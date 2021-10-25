from ..autograd import Tensor, concat

from typing import (
    Union, Tuple, List,
)


def im2col2d(x: Tensor, kernel_shape: Union[Tuple, List], stride: Union[Tuple, List], dilation: Union[Tuple, List]):
    """

    :param x: Tensor (bs, ch_i, h_i, w_i)
    :param kernel_shape: Tuple | List (ch_o, ch_i, h_k, w_k)
    :param stride: Tuple | List ()
    :param dilation: Tuple | List ()
    :return: Tensor (bs, *, i_ch*k_w*k_h)
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
    return concat(ret, dim=1)


def str2pad2d(padding: str, inp_shape, kernel_shape: Union[Tuple, List], stride: Tuple[int, int]):
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
