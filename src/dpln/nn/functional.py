from ..autograd.tensor import Tensor
from ..autograd import function as F

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


def relu(x: Tensor) -> Tensor:
    return F.relu(x)


def sigmoid(x: Tensor) -> Tensor:
    return F.sigmoid(x)


def softmax(x: Tensor, axis=-1) -> Tensor:
    return F.softmax(x, axis=axis)


def tanh(x: Tensor) -> Tensor:
    return F.tanh(x)


def mse_loss(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    # TODO: test
    L = (x - y) ** 2

    if reduction == 'mean':
        return L.mean()
    elif reduction == 'sum':
        return L.sum()
    elif reduction is None:
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or None")


def cross_entropy(x: Tensor, y: Tensor, weight=None, reduction: str = 'mean') -> Tensor:
    # TODO: test
    pass


def l1_loss(x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pass
