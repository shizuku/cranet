import inspect
from src.dpln.autograd.tensor import Tensor
from src.dpln.nn.parameter import Parameter

from typing import (
    Iterator,
    Optional,
)


class Module:
    def parameters(self) -> Iterator[Parameter]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""
        This is typically used to register a buffer that should not to be
        considered a model parameter. 

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.
        """

    def get_name(self) -> str:
        return self.__class__.__name__

    def forward(self, input_: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, input_: Tensor) -> Tensor:
        return self.forward(input_)

    def __repr__(self):
        ...
