
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

