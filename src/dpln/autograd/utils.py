import numpy as np

import math
from typing import (
    Union, Tuple, List, Optional
)


def invert_permutation(permutation):
    if type(permutation) is tuple:
        permutation = list(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def im2col2d(x: np.ndarray,
             kernel_shape: Union[Tuple, List],
             stride: Union[Tuple, List],
             dilation: Union[Tuple, List]) -> np.ndarray:
    """

    :return (bs, ch_i, h_o*w_o, h_k*w_k)
    """
    bs, ch_i, h_i, w_i = x.shape
    _, _, h_k, w_k = kernel_shape
    h_k_t = h_k + (h_k - 1) * (dilation[0] - 1)
    w_k_t = w_k + (w_k - 1) * (dilation[1] - 1)
    ret = []
    for i in range(0, h_i - h_k_t + 1, stride[0]):
        for j in range(0, w_i - w_k_t + 1, stride[1]):
            m = x[:, :, i:i + h_k_t:dilation[0], j:j + w_k_t:dilation[1]]
            ret.append(m.reshape(bs, 1, ch_i, h_k * w_k))
    return np.concatenate(ret, axis=1)


def padding2d(x: np.ndarray,
              padding: Union[List, Tuple],
              mode: str = 'zeros') -> np.ndarray:
    """

    :param x: (bs, ch_i, h_i, w_i)
    :param padding: ((int, int), (int, int))
    :param mode: 'zeros' | 'reflect' | 'replicate' | 'circular'
    :return (bs, ch_i, h_o, w_o)
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
        return np.pad(x, pad_width)
    elif mode == 'reflect':
        return np.pad(x, pad_width, 'reflect')
    elif mode == 'replicate':
        return np.pad(x, pad_width, 'edge')
    elif mode == 'circular':
        return np.pad(x, pad_width, 'symmetric')
    else:
        raise ValueError(
            "value of `mode` must be 'zeros', 'reflect', 'replicate' or 'circular'")


def str2pad2d(padding: str,
              inp_shape: Union[Tuple, List],
              kernel_shape: Union[Tuple, List],
              stride: Tuple[int, int]):
    _, _, h_i, w_i = inp_shape
    _, _, h_k, w_k = kernel_shape
    if padding.lower() == 'same':
        if h_i % stride[0] == 0:
            h_p = max(h_k - stride[0], 0)
        else:
            h_p = max(h_k - (h_i % stride[0]), 0)
        if w_i % stride[1] == 0:
            w_p = max(w_k - stride[1], 0)
        else:
            w_p = max(w_k - (w_i % stride[1]), 0)
        h_pa = h_p // 2
        h_pb = h_p - h_pa
        w_pa = w_p // 2
        w_pb = w_p - w_pa
        return (h_pa, h_pb), (w_pa, w_pb)
    elif padding.lower() == 'valid':
        return (0, 0), (0, 0)
    else:
        raise ValueError("padding string must be 'same' or 'valid'")


def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None,
           stride: Union[Tuple, List, int] = 1,
           padding: Union[Tuple, List, int, str] = 0,
           dilation: Union[Tuple, List, int] = 1,
           groups: int = 1,
           padding_mode: str = 'zeros') -> np.ndarray:
    """

    :param x: (bs, ch_i, h_i, w_i)
    :param weight: (ch_o, ch_i_g, h_k, w_k)
    :param bias: (ch_o)
    :param stride: -> (int, int)
    :param padding: -> ((int, int), (int, int))
    :param dilation: -> (int, int)
    :param groups: int
    :param padding_mode: 'zeros'
    :return (bs, ch_o, h_o, w_o)
    """
    assert x.ndim == 4
    bs, ch_i, h_i, w_i = x.shape
    ch_o, ch_i_g, h_k, w_k = weight.shape
    assert bias is None or bias.ndim == 1

    # make sure stride is (int, int)
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

    # make sure dilation
    dia_err_msg = "value of `dilation` must be tuple of 2 or int"
    if type(dilation) == int:
        dilation = (dilation, dilation)
    elif type(dilation) in [tuple, int]:
        assert len(dilation) == 2, dia_err_msg
    else:
        raise ValueError(dia_err_msg)

    # make sure groups
    assert ch_i % groups == 0 and ch_o % groups == 0
    ch_o_g = ch_o // groups

    h_o = math.floor((h_i + padding[0][0] + padding[0][1] - dilation[0] * (h_k - 1) - 1)
                     / stride[0] + 1)
    w_o = math.floor((w_i + padding[1][0] + padding[1][1] - dilation[1] * (w_k - 1) - 1)
                     / stride[1] + 1)

    pad_x = padding2d(x, padding, padding_mode)
    col_x = im2col2d(pad_x, weight.shape, stride, dilation)
    col_x = col_x.reshape(bs, h_o * w_o, groups, ch_i_g * h_k * w_k)
    col_w = weight.reshape(groups, ch_o_g, ch_i_g * h_k * w_k)
    ret = np.matmul(col_w, col_x, axes=[(1, 2), (3, 1), (2, 3)])
    ret = ret.reshape(bs, ch_o, h_o, w_o)
    if bias is None:
        return ret
    return ret + bias.reshape(1, ch_o, 1, 1)


def rot180_2d(arr: np.ndarray) -> np.ndarray:
    bs, ch, h, w = arr.shape
    new_arr = arr.reshape(bs, ch, h * w)
    new_arr = new_arr[:, :, ::-1]
    new_arr = new_arr.reshape(bs, ch, w, h)
    return new_arr
