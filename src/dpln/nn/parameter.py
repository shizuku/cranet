from ..autograd.tensor import Tensor

from typing import (
    Optional,
)


class Parameter(Tensor):
    def __new__(cls, data: Optional[Tensor] = None, requires_grad: bool = True):
        return super().__init__(data, requires_grad)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
