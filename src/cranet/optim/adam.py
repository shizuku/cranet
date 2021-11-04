import cranet

from .optimizer import Optimizer
from ..nn.parameter import Parameter

import math
from typing import Iterator


class Adam(Optimizer):
    def __init__(self, parameters: Iterator[Parameter],
                 lr: float = 1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, amsgrad=False) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(parameters, defaults)

    @cranet.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with cranet.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = cranet.zeros_like(p)
                    state['exp_avg_sq'] = cranet.zeros_like(p)
                    if amsgrad:
                        state['max_exp_avg_sq'] = cranet.zeros_like(p)

                state['step'] += 1

                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                else:
                    max_exp_avg_sq = None

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if weight_decay != 0:
                    grad += weight_decay * p

                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad

                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * grad * grad.conj()

                if amsgrad:
                    cranet.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (cranet.sqrt(max_exp_avg_sq) / math.sqrt(bias_correction2)) + eps
                else:
                    denom = (cranet.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps
                step_size = lr / bias_correction1
                p += -step_size * (exp_avg / denom)

        return loss
