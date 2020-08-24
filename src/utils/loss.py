
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


