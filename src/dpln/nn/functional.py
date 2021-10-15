from ..autograd.tensor import Tensor, Dependency

import numpy as np

from typing import (
    Optional,
)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = Ax + b`.
    """
    output = x @ weight
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


def l1_loss(x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pass


def mse_loss(x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pass


# TODO somewhere else
def tanh(x: Tensor) -> Tensor:
    data = np.tanh(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
