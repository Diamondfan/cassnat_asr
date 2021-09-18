# 2020 Ruchao Fan

import math
import torch

class BaseOpt(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._step = 0
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
    def __init__(self, optimizer, model_size=512, factor=5.0, warmup_steps=25000, total_steps=40000, warmup_type="noam_warmup"):
        self.factor = factor
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_type = warmup_type
        for p in optimizer.param_groups:
            p['lr'] = factor
        super(NoamOpt, self).__init__(optimizer)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup_type == "noam_warmup":
            return (self.warmup_steps ** 0.5 * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        else:   
            c = self.model_size ** (-0.5)
            if step <= self.warmup_steps:
                return c * step * self.warmup_steps ** (-1.5)
            if self.warmup_type == "custom_exp":
                return c * step ** (-0.5)
            elif self.warmup_type == "custom_linear":
                base_rate = c * self.warmup_steps ** (-0.5)
                decay_num = 1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return base_rate * max(decay_num, 0)

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup_steps": self.warmup_steps,
            "warmup_type": self.warmup_type,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()}

class CosineOpt(BaseOpt):
    "Optim wrapper that implements rate."
    def __init__(self, total, warmup, optimizer):
        self.total = total
        self.warmup = warmup
        super(CosineOpt, self).__init__(optimizer)
        
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

class LRMulStepScheduler(BaseOpt):
    def __init__(self, decay_rate, s_warm, s_decay, s_keep, optimizer):
        self.s_warm = s_warm
        self.s_decay = s_decay
        self.s_keep = s_keep
        self.decay_rate = decay_rate
        super(LRMulStepScheduler, self).__init__(optimizer)

    def rate(self, step = None):
        if step is None:
            step = self._step
        if step <= self.s_warm:
            rate = step / self.s_warm
        elif step <= self.s_decay:
            rate = 1
        elif step <= self.s_keep:
            rate = self.decay_rate ** ((step - self.s_decay) / (self.s_keep - self.s_decay))
        else:
            rate = self.decay_rate
        return rate

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "decay_rate": self.decay_rate, 
            "s_warm": self.s_warm,
            "s_decay": self.s_decay, 
            "s_keep": self.s_keep, 
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()}
            
def get_opt(opt_type, model, args):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    if opt_type == "noam":
        factor = args.noam_factor
        warmup_steps = args.warmup_steps
        total_steps = args.total_steps
        warmup_type = args.warmup_type
        return NoamOpt(opt, args.d_model, factor, warmup_steps, total_steps, warmup_type)
    elif opt_type == "normal":
        return opt
    elif opt_type == "cosine":
        return CosineOpt(args.cosine_total, args.cosine_warmup, opt)
    elif opt_type == "noamwarm":
        return NoamWarmOpt(args.noam_warmup, opt)
    elif opt_type == "multistep":
        return LRMulStepScheduler(args.decay_rate, args.s_warm, args.s_decay, args.s_keep, opt)
    else:
        raise NotImplementedError

