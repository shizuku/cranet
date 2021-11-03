from cranet import Tensor

from .module import Module
from .. import functional as F
from ..parameter import Parameter
from .. import init

import numpy as np

from typing import (
    Optional,
)


class Linear(Module):
    r"""
    Applies a linear transformation to the incoming datasets: :math:`y = xW + b`.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """init(reset) parameters"""
        k = np.sqrt(1 / self.in_features)
        weight_data = init.uniform_([self.in_features, self.out_features], -k, k)
        self.weight = Parameter(weight_data)
        if self.use_bias:
            bias_data = init.uniform_([self.out_features], -k, k)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
