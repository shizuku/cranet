from ..autograd.tensor import Tensor, pad
from ..autograd import function as AF
from . import utils as U

import math
import numpy as np
from typing import (
    Optional,
    Tuple,
    List,
    Union,
)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming datasets: :math:`y = xW + b`.
    """
    output = x @ weight
    if bias is not None:
        return output + bias
    return output


def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Union[Tuple, List, int] = 1,
           padding: Union[Tuple, List, int, str] = 0,
           dilation: Union[Tuple, List, int] = 1,
           groups: int = 1,
           padding_mode: str = 'zeros'):
    """

    :param x: Tensor shape(bs, ch_i, h_i, w_i)
    :param weight: Tensor shape(ch_o, ch_i/groups, h_k, w_k)
    :param bias: Tensor optional shape(ch_o)
    :param stride: int | Tuple | List
    :param padding: int | str | Tuple | List
    :param dilation: int | Tuple | List
    :param groups: int
    :param padding_mode: str 'zeros' | 'reflect' | 'replicate' | 'circular'
    :return: Tensor shape(bs, ch_o, h_o, w_o)
    """
    assert x.dim() == 4
    bs, ch_i, h_i, w_i = x.shape
    ch_o, ch_i_g, h_k, w_k = weight.shape
    assert bias.dim() == 1

    stride_err_msg = "value of `stride` must be tuple of 2 or int"
    if type(stride) == int:
        stride = (stride, stride)
    elif type(stride) in [tuple, list]:
        assert len(stride) == 2, stride_err_msg
    else:
        raise ValueError(stride_err_msg)

    # make sure padding is ((int, int), (int,int))
    pad_err_msg = "value of `padding` must be tuple or list of 2, 2x2 or 4 or int"
    if type(padding) == int:
        padding = ((padding, padding), (padding, padding))
    elif type(padding) in [tuple, list]:
        if len(padding) == 2:
            if type(padding[0]) == int:
                padding = ((padding[0], padding[0]), padding[1])
            elif type(padding[0]) in [tuple, list]:
                assert len(padding[0]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
            if type(padding[1]) == int:
                padding = (padding[0], (padding[1], padding[1]))
            elif type(padding[1]) in [tuple, list]:
                assert len(padding[1]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
        elif len(padding) == 4:
            padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(pad_err_msg)
    elif type(padding) == str:
        padding = U.str2pad2d(padding, x.shape, weight.shape, stride)
    else:
        raise ValueError(pad_err_msg)

    dia_err_msg = "value of `dilation` must be tuple of 2 or int"
    if type(dilation) == int:
        dilation = (dilation, dilation)
    elif type(dilation) in [tuple, int]:
        assert len(dilation) == 2, dia_err_msg
    else:
        raise ValueError(dia_err_msg)

    # make sure groups
    assert ch_i % groups == 0 and ch_o % groups == 0

    x_pad = padding2d(x, padding, padding_mode)

    return AF.conv2d(x_pad, weight, bias, stride=stride, dilation=dilation, groups=groups)


def batch_norm(
        x: Tensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
) -> Tensor:
    _, ch, _, _ = x.shape
    if training:
        b_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        b_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        y = (x - b_mean) / AF.sqrt(b_var + eps)
    else:
        assert running_mean is not None
        assert running_var is not None
        running_mean = running_mean.reshape(1, ch, 1, 1)
        running_var = running_var.reshape(1, ch, 1, 1)
        y = (x - running_mean) / AF.sqrt(running_var + eps)
    if weight is not None and bias is not None:
        y = y * weight.reshape(1, ch, 1, 1) + bias.reshape(1, ch, 1, 1)
    return y


def padding2d(x: Tensor, padding: Union[Tuple, List, int], mode: str) -> Tensor:
    """

    :param x: Tensor (bs, ch_i, h_i, w_i)
    :param padding: int | Tuple | List
    :param mode: str 'zeros' | 'reflect' | 'replicate' | 'circular'
    :return: Tensor (bs, ch_i, h_i+2*p[0], w_i+2*p[1])
    """
    # make sure padding is ((int, int), (int,int))
    pad_err_msg = "value of `padding` must be tuple or list of 2, 2x2 or 4 or int"
    if type(padding) == int:
        padding = ((padding, padding), (padding, padding))
    elif type(padding) in [tuple, list]:
        if len(padding) == 2:
            if type(padding[0]) == int:
                padding = ((padding[0], padding[0]), padding[1])
            elif type(padding[0]) in [tuple, list]:
                assert len(padding[0]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
            if type(padding[1]) == int:
                padding = (padding[0], (padding[1], padding[1]))
            elif type(padding[1]) in [tuple, list]:
                assert len(padding[1]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
        elif len(padding) == 4:
            padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(pad_err_msg)
    else:
        raise ValueError(pad_err_msg)

    pad_width = ((0, 0), (0, 0), *padding)
    if mode == 'zeros':
        return pad(x, pad_width)
    elif mode == 'reflect':
        return pad(x, pad_width, 'reflect')
    elif mode == 'replicate':
        return pad(x, pad_width, 'edge')
    elif mode == 'circular':
        return pad(x, pad_width, 'symmetric')
    else:
        raise ValueError("value of `mode` must be 'zeros', 'reflect', 'replicate' or 'circular'")


def max_pool2d(x: Tensor, kernel_size: Union[Tuple, List, int],
               stride: Optional[Union[Tuple, List, int]] = None,
               padding: Union[Tuple, List, int] = 0,
               dilation: Union[Tuple, List, int] = 1,
               padding_mode: str = 'zeros') -> Tensor:
    assert x.dim() == 4

    bs, ch_i, h_i, w_i = x.shape
    if type(kernel_size) == int:
        kernel_size = (ch_i, ch_i, kernel_size, kernel_size)
    elif type(kernel_size) in [tuple, list]:
        kernel_size = (ch_i, ch_i, *kernel_size)
    _, _, h_k, w_k = kernel_size

    stride_err_msg = "value of `stride` must be tuple of 2 or int"
    if stride is None:
        stride = (kernel_size[2], kernel_size[3])
    elif type(stride) == int:
        stride = (stride, stride)
    elif type(stride) in [tuple, list]:
        assert len(stride) == 2, stride_err_msg
    else:
        raise ValueError(stride_err_msg)

    # make sure padding is ((int, int), (int,int))
    pad_err_msg = "value of `padding` must be tuple or list of 2, 2x2 or 4 or int"
    if type(padding) == int:
        padding = ((padding, padding), (padding, padding))
    elif type(padding) in [tuple, list]:
        if len(padding) == 2:
            if type(padding[0]) == int:
                padding = ((padding[0], padding[0]), padding[1])
            elif type(padding[0]) in [tuple, list]:
                assert len(padding[0]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
            if type(padding[1]) == int:
                padding = (padding[0], (padding[1], padding[1]))
            elif type(padding[1]) in [tuple, list]:
                assert len(padding[1]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
        elif len(padding) == 4:
            padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(pad_err_msg)
    else:
        raise ValueError(pad_err_msg)

    dia_err_msg = "value of `dilation` must be tuple of 2 or int"
    if type(dilation) == int:
        dilation = (dilation, dilation)
    elif type(dilation) in [tuple, int]:
        assert len(dilation) == 2, dia_err_msg
    else:
        raise ValueError(dia_err_msg)

    h_o = math.floor((h_i + padding[0][0] + padding[0][1] - dilation[0] * (h_k - 1) - 1) / stride[0] + 1)
    w_o = math.floor((w_i + padding[1][0] + padding[1][1] - dilation[1] * (w_k - 1) - 1) / stride[1] + 1)

    pad_inp = padding2d(x, padding, padding_mode)
    col = U.im2col2d(pad_inp, kernel_size, stride, dilation)
    a = AF.max(col, dim=-1).reshape(bs, h_o, w_o, ch_i)
    return a.permute(dims=(0, 3, 1, 2))


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return AF.dropout(x, p, training)


def flatten(x: Tensor, start_dim: int = 1, end_dim: int = -1) -> Tensor:
    return x.flatten(start_dim, end_dim)


# activation

def relu(x: Tensor) -> Tensor:
    return AF.relu(x)


def sigmoid(x: Tensor) -> Tensor:
    return AF.sigmoid(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return AF.softmax(x, dim=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return AF.log(AF.softmax(x, dim=dim))


def tanh(x: Tensor) -> Tensor:
    return AF.tanh(x)


# loss

def l1_loss(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    L = AF.abs(x - y)

    if reduction.lower() == 'mean':
        return L.mean()
    elif reduction.lower() == 'sum':
        return L.sum()
    elif reduction.lower() == 'none':
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or None")


def mse_loss(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    L = (x - y) ** Tensor(2)

    if reduction.lower() == 'mean':
        return L.mean()
    elif reduction.lower() == 'sum':
        return L.sum()
    elif reduction.lower() == 'none':
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or None")


def binary_cross_entropy(x: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    """
    Args:
        x: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        y: :math:`(N, *)`, same shape as the input
        weight: a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size N.
        reduction:

    Returns:
        scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as input.
    """
    assert np.logical_and(x.data > 0, x.data < 1).all(), "elements of input should between 0 and 1"
    w = Tensor(1)
    if weight:
        w = weight

    L = -w * (y * AF.log(x) + (1 - y) * AF.log(1 - x))

    if reduction == 'mean':
        return L.mean()
    elif reduction == 'sum':
        return L.sum()
    elif reduction.lower() == 'none':
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or None")


def nll_loss(x: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    # TODO: impl weight
    assert x.dim() == 2
    assert y.dim() == 1

    w = Tensor(1)
    if weight:
        assert weight.dim() == 1
        w = weight

    L = -w * AF.nll(x, y, dim=-1)

    if reduction.lower() == 'mean':
        return L.mean()
    elif reduction.lower() == 'sum':
        return L.sum()
    elif reduction.lower() == 'none':
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def cross_entropy(x: Tensor, y: Tensor, weight=None, reduction: str = 'mean') -> Tensor:
    # TODO: impl weight
    return nll_loss(log_softmax(x), y, weight, reduction)
