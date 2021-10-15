import numpy as np
from numpy.testing._private.utils import requires_memory
from ..autograd.tensor import Dependency, Tensor

from typing import (
    Optional,
)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = Ax + b`.
    """
    output = input.matmul(weight.t())
    if bias is None:
        output += bias
    ret = output
    return ret

def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`
    """
    ...


def l1_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pass


def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pass


# TODO somewhere else
def tanh(input: Tensor) -> Tensor:
    data = np.tanh(input.data)
    requires_grad = input.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)
        
        depends_on = [Dependency(input, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)