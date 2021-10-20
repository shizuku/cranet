from ..autograd.tensor import Tensor
from ..autograd import function as F
from .utils import im2col2d

import math
from typing import (
    Optional,
)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming datasets: :math:`y = xW + b`.
    """
    output = x @ weight
    if bias is not None:
        return output + bias
    return output


def conv2d(inp: Tensor, weight: Tensor, bias: Tensor, stride=1, padding=0, dilation=1, groups=1):
    """
    TODO: impl padding, dilation, groups
    :param inp: Tensor shape(bs, ch_i, h_i, w_i)
    :param weight: Tensor shape(ch_o, ch_i, h_k, w_k)
    :param bias: Tensor shape(ch_o)
    :param stride: int | Tuple[int, int]
    :param padding: int | Tuple[int, int]
    :param dilation: int | Tuple[int, int]
    :param groups: int
    :return: Tensor shape(bs, ch_o, h_o, w_o)
    """
    if type(stride) == int:
        stride = (stride, stride)
    if type(padding) == int:
        padding = (padding, padding)
    if type(dilation) == int:
        dilation = (dilation, dilation)
    bs, _, h_i, w_i = inp.shape
    ch_o, ch_i, h_k, w_k = weight.shape
    h_o = math.floor((h_i + 2 * padding[0] - dilation[0] * (h_k - 1) - 1) / stride[0] + 1)
    w_o = math.floor((w_i + 2 * padding[1] - dilation[1] * (w_k - 1) - 1) / stride[1] + 1)
    col = im2col2d(inp, weight.shape, stride)
    return (weight.reshape(ch_o, ch_i * h_k * w_k) @ col.transpose(1, 2)).reshape(bs, ch_o, h_o, w_o) + bias.reshape(1, ch_o, 1, 1)


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
