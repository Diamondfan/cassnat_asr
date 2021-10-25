
import torch
import torch.nn as nn

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
    def __init__(self, padding_idx):
        super(KLDivLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.padding_idx = padding_idx
        
    def forward(self, x, at_prob, target):
        """
        x: bU x class
        taget: bU x class
        """
        assert x.size(1) == at_prob.size(1)
        mask = target != self.padding_idx
        tokens = mask.sum().item()
        return self.criterion(x, at_prob).masked_fill(mask.unsqueeze(1)==0, 0).sum() / tokens

class TPLoss(nn.Module):
    def __init__(self):
        super(TPLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, tp_out, aligned_seq, src_mask):
        """tp_out: b * T * d, aligned_seq: b * T src_mask: b * 1 * T"""
        tp_loss = self.criterion(tp_out.view(-1, tp_out.size(-1)), aligned_seq.view(-1))
        tokens = src_mask.sum().item()
        tp_loss = tp_loss.masked_fill(src_mask.squeeze(1).reshape(-1)==0, 0).sum() / tokens
        acc = (torch.argmax(tp_out, -1) == aligned_seq).masked_fill(src_mask.squeeze(1)==0, 0).sum().item()
        return tp_loss, acc, tokens

class EmbedLoss(nn.Module):
    def __init__(self, padding_idx, embed_loss_type='l1'):
        super(EmbedLoss, self).__init__()
        self.padding_idx = padding_idx
        self.loss_type = embed_loss_type # ce, regression, contrasitive
        if embed_loss_type == 'l2':
            self.criterion = torch.nn.MSELoss(reduction='none')
        elif embed_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif embed_loss_type == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        elif embed_loss_type == 'contrast':
            self.criterion = 1
        else:
            raise NotImplementedError

    def forward(self, pred_embed, word_embed, tgt, tgt_mask):
        if self.loss_type == 'l2':
            true_embed = word_embed(tgt)
            tgt_mask = tgt_mask.transpose(1,2).reshape(-1, 1)
            tokens = tgt_mask.sum().item()
            loss = self.criterion(pred_embed.view(-1, pred_embed.size(-1)), true_embed.view(-1, true_embed.size(-1)))
            return loss.masked_fill(tgt_mask==0, 0).sum() / tokens / loss.size(1)

class ContrastiveLoss():
    def __init__():
        pass

class HubertLoss():
    def __init__():
        pass

class PredictLoss(nn.Module):
    def __init__(self, k_shift, n_generator, loss_type):
        super(PredictLoss, self).__init__()
        self.k_shift = k_shift
        self.n_generator = n_generator
        self.loss_type = loss_type
        if self.loss_type == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif self.loss_type == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError

    def forward(self, nn_out, tgt):
        # nn_out: k, bs x seq_len x encoded_size
        # tgt: bs x seq_len x encoded_size

        seq_len = tgt.size(1) - 1
        seq_len = seq_len if seq_len % 4 == 0 else seq_len - (seq_len % 4)
        bs, _, encoded_size = nn_out[0].size()

        loss = 0
        for i in range(self.n_generator):
            start = (self.k_shift + i - 1) * 4 + 1
            tgt_i = tgt[:,start: seq_len+1,:].reshape(bs, -1, encoded_size)
            used_seq_len = tgt_i.size(1)
            pred = nn_out[i][:,:used_seq_len,:]
            mask = (tgt_i != 0).reshape(-1, encoded_size)
            loss += self.criterion(pred.reshape(-1, encoded_size), tgt_i.reshape(-1, encoded_size)).masked_fill(mask==0, 0).sum() / mask.sum()
        return loss /  self.n_generator


