from .module import Module
from .. import functional as F
from ..parameter import Parameter
from src.dpln.autograd.tensor import Tensor

from typing import (
    Optional,
)


class Linear(Module):
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, require_bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter()
        if require_bias:
            self.bias = Parameter()
        else:
            self.bias = None
        self.reset_parameters()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """add a parameter to the module"""
        ...

    def reset_parameters(self) -> None:
        """init(reset) parameters"""
        ...

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Bilinear(Module):
    ...
