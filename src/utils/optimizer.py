# 2020 Ruchao Fan

import math
import torch

class BaseOpt(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._step = 0
        for p in self.optimizer.param_groups:
            p['initial_lr'] = p['lr']

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = p['initial_lr'] * rate
        self.optimizer.step()

    def rate(self, step=None):
        raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, rank, use_cuda):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(rank)
            else:
                setattr(self, key, value)

class MulBaseOpt(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._step = 0 
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['initial_lr'] = self.optimizer.param_groups[i]['lr']

    def step(self):
        self._step += 1
        rate = self.rate()
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['initial_lr'] * rate[i]
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
        state_dict = {
            "_step": self._step, "warmup_steps": self.warmup_steps,
            "warmup_type": self.warmup_type, "factor": self.factor,
            "model_size": self.model_size, "total_steps": self.total_steps,
            "optimizer": self.optimizer.state_dict()}
        return state_dict

class MulNoamOpt(MulBaseOpt):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, factor, pretrained_factor, warmup_steps, pretrained_warmup_steps, pretrained_idx, total_steps=40000, warmup_type="noam_warmup"):
        self.optimizer = optimizer
        self.factor = factor
        self.pretrained_factor = pretrained_factor
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.pretrained_warmup_steps = pretrained_warmup_steps
        self.pretrained_idx = pretrained_idx
        self.total_steps = total_steps
        self.warmup_type = warmup_type
        self.n_groups = len(optimizer.param_groups)
        for i in range(self.n_groups):
            if i >= pretrained_idx:
                optimizer.param_groups[i]['lr'] = factor
            else:
                optimizer.param_groups[i]['lr'] = pretrained_factor
        super(MulNoamOpt, self).__init__(optimizer)

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup_type == "noam_warmup":
            rate = (self.warmup_steps ** 0.5 * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
            rate_pretrained = (self.pretrained_warmup_steps ** 0.5 * min(step ** (-0.5), step * self.pretrained_warmup_steps ** (-1.5)))
        else:
            c = self.model_size ** (-0.5)
            if step <= self.warmup_steps:
                rate = c * step * self.warmup_steps ** (-1.5)
            else:
                if self.warmup_type == "custom_exp":
                    rate = c * step ** (-0.5)
                elif self.warmup_type == "custom_linear":
                    base_rate = c * self.warmup_steps ** (-0.5)
                    decay_num = 1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                    rate = base_rate * max(decay_num, 0)

            if step <= self.pretrained_warmup_steps:
                rate_pretrained = c * step * self.pretrained_warmup_steps ** (-1.5)
            else:
                if self.warmup_type == "custom_exp":
                    rate_pretrained = c * step ** (-0.5)
                elif self.warmup_type == "custom_linear":
                    base_rate = c * self.pretrained_warmup_steps ** (-0.5)
                    decay_num = 1 - (step - self.pretrained_warmup_steps) / (self.total_steps - self.pretrained_warmup_steps)
                    rate_pretrained = base_rate * max(decay_num, 0)

        rates = []
        for i in range(self.n_groups):
            if  i >= self.pretrained_idx:
                rates.append(rate)
            else:
                rates.append(rate_pretrained)
        return rates

    def state_dict(self):
        state_dict = {
            "_step": self._step, "factor": self.factor,
            "pretrained_factor": self.pretrained_factor, 
            "warmup_steps": self.warmup_steps, "warmup_type": self.warmup_type,
            "pretrained_warmup_steps": self.pretrained_warmup_steps, 
            "pretrained_idx": self.pretrained_idx, "total_steps": self.total_steps, 
            "model_size": self.model_size, "n_groups": self.n_groups,
            "optimizer": self.optimizer.state_dict()}
        return state_dict

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
        state_dict =  {
            "_step": self._step, "warmup": self.warmup, "total": self.total,
            "optimizer": self.optimizer.state_dict()}
        return state_dict

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
        state_dict = {
            "_step": self._step, "decay_rate": self.decay_rate, 
            "s_warm": self.s_warm, "s_decay": self.s_decay, 
            "s_keep": self.s_keep, "optimizer": self.optimizer.state_dict()}
        return state_dict
            
def get_optim(optim_type, model, args):
    eps = args.eps if hasattr(args, 'eps') else 1e-9
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, betas=(0.9, 0.98), eps=eps, weight_decay=args.weight_decay)
    if optim_type == "noam":
        factor = args.noam_factor
        warmup_steps = args.warmup_steps
        total_steps = args.total_steps
        warmup_type = args.warmup_type
        return NoamOpt(optimizer, args.d_model, factor, warmup_steps, total_steps, warmup_type)
    
    if optim_type == "normal":
        return optimizer
    
    if optim_type == "cosine":
        return CosineOpt(args.cosine_total, args.cosine_warmup, optimizer)
     
    if optim_type == "multistep":
        return LRMulStepScheduler(args.decay_rate, args.s_warm, args.s_decay, args.s_keep, optimizer)
    
    raise NotImplementedError


def get_ctc_mul_opt(opt_type, model, args):
    updated_params = [{"params": model.src_embed.parameters()}, {"params": model.encoder.parameters()}, 
                        {"params": model.ctc_generator.parameters(), "lr": args.learning_rate} ]
    if args.interctc_alpha > 0:
        updated_params.append({"params": model.interctc_generator.parameters(), "lr": args.learning_rate})

    opt = torch.optim.Adam(updated_params, lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    if opt_type == "noam":
        factor = args.noam_factor
        pretrained_factor = args.pretrained_noam_factor
        warmup_steps = args.warmup_steps
        pretrained_warmup_steps = args.pretrained_warmup_steps
        total_steps = args.total_steps
        warmup_type = args.warmup_type
        pretrained_idx = 2 
        return MulNoamOpt(opt, args.d_model, factor, pretrained_factor, warmup_steps, pretrained_warmup_steps, pretrained_idx, total_steps, warmup_type)
    elif opt_type == "normal":
        return opt 
    elif opt_type == "cosine":
        return CosineOpt(args.cosine_total, args.cosine_warmup, opt)
    elif opt_type == "multistep":
        return LRMulStepScheduler(args.decay_rate, args.s_warm, args.s_decay, args.s_keep, opt)
    else:
        raise NotImplementedError

