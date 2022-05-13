
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def set_smoothing(self, smoothing):
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        """
        x: bU x class
        taget: bU
        """
        assert x.size(1) == self.size
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            mask = target != self.padding_idx 
            tokens = mask.sum().item()
            target = target.masked_fill(mask==0, 0)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            self.true_dist = true_dist
        return self.criterion(x, true_dist).masked_fill(mask.unsqueeze(1)==0, 0).sum() / tokens

    def MWER(self, att_out, ctc_target, wer_weight):
        """
        att_out: bs x U x vocab, ctc_target: bs x U, wer_weight: bs, norm_prob: b x sample_num
        """
        tgt_mask = ctc_target != 0
        #batch_size, sample_num = norm_prob.size()
        tokens = torch.sum(tgt_mask, 1).reshape(wer_weight.size())
        att_prob = att_out.gather(-1, ctc_target.unsqueeze(-1)).squeeze(-1).masked_fill(tgt_mask==0, 0).sum(1).reshape(wer_weight.size()) / tokens.float()
        wer_weight = wer_weight.float()
        wer_weight = wer_weight - wer_weight.max(1, keepdim=True)[0]
        wer_loss = (att_prob * wer_weight) #* norm_prob
        wer_loss = torch.mean(wer_loss, 1).mean()
        return wer_loss

    def forward_best_path(self, x, target, tgt_mask_pred):
        "x: b x U x class"
        assert x.size(2) == self.size
        preserve = min(x.size(1), target.size(1))
        x, target = x[:,:preserve,:].reshape(-1, self.size), target[:,:preserve].reshape(-1)
        tgt_mask_pred = tgt_mask_pred.squeeze(1)[:,:preserve].reshape(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            mask = target != self.padding_idx 
            tokens = mask.sum().item()
            target = target.masked_fill(mask==0, 0)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            self.true_dist = true_dist
            mask = mask.long() & tgt_mask_pred
        return self.criterion(x, true_dist).masked_fill(mask.unsqueeze(1)==0, 0).sum() / tokens

class KLDivLoss(nn.Module):
    def __init__(self, padding_idx, kd_weight=0.1):
        super(KLDivLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.padding_idx = padding_idx
        self.kd_weight = kd_weight
        
    def forward(self, x, at_prob, target):
        """
        x: bU x class
        taget: bU x class
        """
        assert x.size(1) == at_prob.size(1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(0)
            mask = target != self.padding_idx 
            tokens = mask.sum().item()
            target = target.masked_fill(mask==0, 0)
            true_dist.scatter_(1, target.unsqueeze(1), 1)
            target_dist = (1 - self.kd_weight) * true_dist + self.kd_weight * at_prob
        return self.criterion(x, target_dist).masked_fill(mask.unsqueeze(1)==0, 0).sum() / tokens

class Wav2vecLoss(nn.Module):
    def __init__(self, infonce=False, loss_weights=None, log_keys=None):
        super(Wav2vecLoss, self).__init__()
        self.infonce = infonce
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, net_output, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        logits = model.get_logits(net_output).float()
        target = model.get_targets(net_output)

        weights = None

        losses = []

        reduction = "none" if (not reduce) else "sum"
        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )

        #sample_size = sample["net_input"]["mask_indices"].sum()
        sample_size = target.numel() if self.infonce else target.long().sum().item()
        loss /= sample_size
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() #* sample_size
                    loss += p
                    losses.append(p)
                else:
                    losses.append(torch.Tensor([0]))

        logging_output = {
            "loss": loss.item() if (reduce) else loss.detach(),
            "ntokens": sample_size,
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(sample, net_output)
                    else:
                        original_target = target
                    logging_output["target"] = original_target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output

