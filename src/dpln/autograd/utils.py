import numpy as np

from typing import (
    Union, Tuple, List,
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

    :return (bs, h_o*w_o, ch_i, h_k*w_k)
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


def rot180_2d(arr: np.ndarray) -> np.ndarray:
    bs, ch, h, w = arr.shape
    new_arr = arr.reshape((bs, ch, h * w))
    new_arr = new_arr[:, :, ::-1]
    new_arr = new_arr.reshape((bs, ch, w, h))
    return new_arr


def unstride(x: np.ndarray, stride_i) -> np.ndarray:
    bs, ch_o, h_o, w_o = x.shape
    h_st, w_st = stride_i
    if h_st == 1 and w_st == 1:
        return x
    else:
        # h_o_s = (h_o - 1) * h_st + 1
        # w_o_s = (w_o - 1) * w_st + 1
        h_o_s = h_o * h_st
        w_o_s = w_o * w_st
        x_s = np.zeros((bs, ch_o, h_o_s, w_o_s))
        for i in range(h_o):
            for j in range(w_o):
                x_s[:, :, i * h_st, j * w_st] = x[:, :, i, j]
        return x_s


def unpadding(x: np.ndarray, kernel_size) -> np.ndarray:
    _, _, h_k, w_k = kernel_size
    padding = ((h_k - 1, h_k - 1),
               (w_k - 1, w_k - 1),)
    x_p = padding2d(x, padding)
    return x_p


def pad_stride(x: np.ndarray, w_r_shape, stride_i) -> np.ndarray:
    """

    :param x: (bs, ch_o, h_o, w_o)
    :param w_r_shape: (bs, ch_i, h_k, w_k)
    :param stride_i: (int, int)
    :return:
    """
    bs, ch_o, h_o, w_o = x.shape
    _, _, h_k, w_k = w_r_shape
    h_st, w_st = stride_i

    if h_st == 1 and w_st == 1:
        x_s = x
    else:
        # h_o_s = (h_o - 1) * h_st + 1
        # w_o_s = (w_o - 1) * w_st + 1
        h_o_s = h_o * h_st
        w_o_s = w_o * w_st
        x_s = np.zeros((bs, ch_o, h_o_s, w_o_s))
        for i in range(h_o):
            for j in range(w_o):
                x_s[:, :, i * h_st, j * w_st] = x[:, :, i, j]

    padding = ((h_k - 1, h_k - 1),
               (w_k - 1, w_k - 1),)
    x_s_p = padding2d(x_s, padding)
    return x_s_p
