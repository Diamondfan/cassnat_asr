# 2020 Ruchao Fan

import math
import torch

class BaseOpt(object):
    def __init__(self, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 0
        for p in self.optimizer.param_groups:
            p['initial_lr'] = p['lr']

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = p['initial_lr'] * rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)

class NoamOpt(BaseOpt):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.factor = factor
        self.model_size = model_size
        for p in optimizer.param_groups:
            p['lr'] = factor
        super(NoamOpt, self).__init__(warmup, optimizer)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
                 min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()}

class NoamWarmOpt(BaseOpt):
    "Optim wrapper that implements rate."
    def __init__(self, warmup, optimizer):
        super(NoamWarmOpt, self).__init__(warmup, optimizer)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.warmup ** 0.5 *
                 min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()}


class CosineOpt(BaseOpt):
    "Optim wrapper that implements rate."
    def __init__(self, total, warmup, optimizer):
        self.total = total
        super(CosineOpt, self).__init__(warmup, optimizer)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0.5 * (math.cos(math.pi * (step - self.warmup) / self.total) + 1)

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "total": self.total,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()}

            
def get_opt(opt_type, model, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    if opt_type == "noam":
        return NoamOpt(args.d_model, args.noam_factor, args.noam_warmup, opt)
    elif opt_type == "normal":
        return opt
    elif opt_type == "cosine":
        return CosineOpt(args.cosine_total, args.cosine_warmup, opt)
    elif opt_typpe == "noamwarm":
        return NoamWarmOpt(args.noam_warmup, opt)
    else:
        raise NotImplementedError
                


