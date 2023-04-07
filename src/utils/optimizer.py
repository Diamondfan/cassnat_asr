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
    def __init__(self, optimizer, freeze_steps=None):
        self.optimizer = optimizer
        self._step = 0
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['initial_lr'] = self.optimizer.param_groups[i]['lr']
        self.freeze_steps = freeze_steps

    def step(self, step_optimizer=True):
        self._step += 1
        rate = self.rate()

        assert len(rate) == len(self.freeze_steps)
        for i in range(len(self.freeze_steps)):
            if self._step <= self.freeze_steps[i]:
                rate[i] = 0

        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['initial_lr'] * rate[i]

        if step_optimizer:
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
    def __init__(self, optimizer, model_size, factors, warmup_steps, freeze_steps, total_steps=40000, warmup_type="noam_warmup"):
        # factors: list   warmup_steps: list 
        self.optimizer = optimizer
        self.factors = factors
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_type = warmup_type
        self.n_groups = len(optimizer.param_groups)
        for i in range(self.n_groups):
            optimizer.param_groups[i]['lr'] = factors[i]
        super(MulNoamOpt, self).__init__(optimizer, freeze_steps)

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        rates = []
        for i in range(self.n_groups):
            rates.append(0)
            if self.freeze_steps[i] > 0 and step > self.freeze_steps[i]:
                cur_step = step - self.freeze_steps[i]
            elif self.freeze_steps[i] > 0 and step <= self.freeze_steps[i]:
                continue
            else:
                cur_step = step
            if self.warmup_type == "noam_warmup":
                rates[i] = (self.warmup_steps[i] ** 0.5 * min(cur_step ** (-0.5), cur_step * self.warmup_steps[i] ** (-1.5)))
            else:
                c = self.model_size ** (-0.5)
                if cur_step <= self.warmup_steps[i]:
                    rates[i] = c * cur_step * self.warmup_steps[i] ** (-1.5)
                else:
                    if self.warmup_type == "custom_exp":
                        rates[i] = c * cur_step ** (-0.5)
                    elif self.warmup_type == "custom_linear":
                        base_rate = c * self.warmup_steps[i] ** (-0.5)
                        decay_num = 1 - (cur_step - self.warmup_steps[i]) / (self.total_steps - self.warmup_steps[i])
                        rates[i] = base_rate * max(decay_num, 0)
        return rates

    def state_dict(self):
        state_dict = {
            "_step": self._step, "factors": self.factors,
            "warmup_steps": self.warmup_steps, "warmup_type": self.warmup_type,
            "total_steps": self.total_steps, "model_size": self.model_size, 
            "n_groups": self.n_groups, "optimizer": self.optimizer.state_dict()}
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

def get_mul_optim(optim_type, update_params_group, args):
    eps = args.eps if hasattr(args, 'eps') else 1e-9
    updated_params = []
    for param in update_params_group:
        updated_params.append({"params": param } )

    optimizer = torch.optim.Adam(updated_params, lr=args.learning_rate, betas=(0.9, 0.98), eps=eps, weight_decay=args.weight_decay)
    assert optim_type == "noam", "support and have tested noam only currently"
    factors = args.noam_factor
    warmup_steps = args.warmup_steps
    freeze_steps = args.freeze_steps
    total_steps = args.total_steps
    warmup_type = args.warmup_type

    assert len(update_params_group) == len(factors)
    return MulNoamOpt(optimizer, args.d_model, factors, warmup_steps, freeze_steps, total_steps, warmup_type)

