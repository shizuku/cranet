from ..autograd.tensor import Tensor, Arrayable, Dependency

from typing import (
    List,
)


class Parameter(Tensor):
    def __init__(self, data: Arrayable = None, requires_grad: bool = True, dependencies: List[Dependency] = None):
        super().__init__(data, requires_grad, dependencies)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
