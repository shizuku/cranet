from ..autograd.tensor import Tensor, Tensorable, Dependency

from typing import (
    List,
)


class Parameter(Tensor):
    def __init__(self, data: Tensorable, requires_grad: bool = True, dependencies: List[Dependency] = None, dtype=None):
        super().__init__(data, requires_grad, dependencies, dtype=dtype)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()
