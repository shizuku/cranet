from ..nn.parameter import Parameter

from typing import Iterator


class Optimizer:
    def __init__(self, parameters: Iterator[Parameter], *args, **kwargs) -> None:
        self.parameters = list(parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self, closure) -> None:
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError
