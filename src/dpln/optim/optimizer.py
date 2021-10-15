from ..nn.modules.module import Module


class Optimizer:

    def __init__(self, module: Module, lr: float) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        if not self.module:
            raise AttributeError(
                "'NoneType' object has no attribute 'zero_grad'")
        for parameter in self.module.parameters():
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
