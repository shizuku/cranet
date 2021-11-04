import cranet

from .optimizer import Optimizer, required
from ..nn.parameter import Parameter

from typing import Iterator, Optional


class SGD(Optimizer):
    def __init__(self, parameters: Iterator[Parameter], lr: Optional[float] = required,
                 momentum: Optional[float] = 0, dampening: Optional[float] = 0,
                 weight_decay: Optional[float] = 0, nesterov: Optional[bool] = False
                 ) -> None:

        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        super().__init__(parameters, defaults)

    @cranet.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with cranet.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = cranet.zeros_like(p)

                state['step'] += 1

                step = state['step']
                exp_avg = state['exp_avg']

                if weight_decay != 0:
                    grad += weight_decay * p

                if momentum != 0:
                    if step > 1:
                        exp_avg *= momentum
                        exp_avg += (1 - dampening) * grad
                    else:
                        exp_avg = grad
                    if nesterov:
                        grad += momentum * exp_avg
                    else:
                        grad = momentum
                p += - lr * grad
        return loss
