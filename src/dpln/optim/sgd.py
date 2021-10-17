from .optimizer import Optimizer
from ..nn.parameter import Parameter

from typing import Iterator


class SGD(Optimizer):
    def __init__(self, parameters: Iterator[Parameter], lr: float = 0.01, *args, **kwargs) -> None:
        super().__init__(parameters, *args, **kwargs)
        self.lr = lr

    def step(self, **kwargs) -> None:
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr
