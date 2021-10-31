# TODO: impl Hebb
from cranet.nn.modules import Module
from .optimizer import Optimizer
from ..nn.parameter import Parameter

from typing import Iterator


class Oja(Optimizer):
    def __init__(self, parameters: Iterator[Parameter], lr: float = 0.01, *args, **kwargs) -> None:
        super().__init__(parameters, *args, **kwargs)
        self.lr = lr

    def step(self, model: Module) -> None:
        for parameter in self.parameters:
            parameter = parameter + self.lr
