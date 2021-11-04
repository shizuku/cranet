import cranet

from .optimizer import Optimizer, required
from ..nn.parameter import Parameter

from typing import Iterator


class SGD(Optimizer):
    def __init__(self, parameters: Iterator[Parameter], lr: float = required) -> None:
        defaults = dict(lr=lr)
        super().__init__(parameters, defaults)

    @cranet.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with cranet.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                p -= p.grad * lr
        return loss
