from ..autograd.tensor import Tensor, pad, concat
from ..autograd import function as AF
from .utils import im2col2d, str2pad2d

import math
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
    bs, _, h_i, w_i = x.shape
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
        padding = str2pad2d(padding, x.shape, weight.shape, stride)
    else:
        raise ValueError(pad_err_msg)

    dia_err_msg = "value of `dilation` must be tuple of 2 or int"
    if type(dilation) == int:
        dilation = (dilation, dilation)
    elif type(dilation) in [tuple, int]:
        assert len(dilation) == 2, dia_err_msg
    else:
        raise ValueError(dia_err_msg)

    assert ch_o % groups == 0
    ch_i = ch_i_g * groups

    h_o = math.floor((h_i + padding[0][0] + padding[0][1] - dilation[0] * (h_k - 1) - 1) / stride[0] + 1)
    w_o = math.floor((w_i + padding[1][0] + padding[1][1] - dilation[1] * (w_k - 1) - 1) / stride[1] + 1)

    pad_inp = padding2d(x, padding, padding_mode)
    col = im2col2d(pad_inp, weight.shape, stride, dilation)
    col = col.reshape(bs, h_o * w_o, groups, ch_i_g * w_k * h_k)  # (bs, h_o*w_o, groups, ch_i_g, w_k*h_k)
    g = []
    ch_o_g = ch_o // groups
    for i in range(groups):
        a_w = weight[i * ch_o_g:(i + 1) * ch_o_g, :, :, :]
        a_w = a_w.reshape(ch_o_g, ch_i_g * h_k * w_k)
        a_x = col[:, :, i, :]
        a_x = a_x.transpose(1, 2)
        a = (a_w @ a_x)
        g.append(a)

    ret = concat(g, axis=1)
    ret = ret.reshape(bs, ch_o, h_o, w_o)
    if bias is None:
        return ret
    return ret + bias.reshape(1, ch_o, 1, 1)


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
               stride: Union[Tuple, List, int] = 1,
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
    col = im2col2d(pad_inp, kernel_size, stride, dilation)
    a = AF.max(col, axis=-1).reshape(bs, h_o, w_o, ch_i)
    return a.permute(axes=(0, 3, 1, 2))


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return AF.dropout(x, p, training)


def flatten(x: Tensor, start_dim=1, end_dim=-1) -> Tensor:
    return x.flatten(start_dim, end_dim)


# activation

def relu(x: Tensor) -> Tensor:
    return AF.relu(x)


def sigmoid(x: Tensor) -> Tensor:
    return AF.sigmoid(x)


def softmax(x: Tensor, axis=-1) -> Tensor:
    return AF.softmax(x, axis=axis)


def tanh(x: Tensor) -> Tensor:
    return AF.tanh(x)


# loss

def l1_loss(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    # TODO: test
    L = AF.abs(x - y)

    if reduction == 'mean':
        return L.mean()
    elif reduction == 'sum':
        return L.sum()
    elif reduction is None:
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or None")


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
