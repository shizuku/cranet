from ..autograd import Tensor, concat

from typing import (
    Union, Tuple, List,
)


def im2col2d(inp: Tensor, kernel_shape: Union[Tuple, List], stride: Union[Tuple, List], dilation: Union[Tuple, List]):
    """

    :param inp: Tensor (bs, ch_i, h_i, w_i)
    :param kernel_shape: Tuple | List (ch_o, ch_i, h_k, w_k)
    :param stride: Tuple | List ()
    :param dilation: Tuple | List ()
    :return: Tensor (bs, *, i_ch*k_w*k_h)
    """
    bs, _, h_i, w_i = inp.shape
    _, ch_i, h_k, w_k = kernel_shape
    h_k_t = h_k + (h_k - 1) * (dilation[0] - 1)
    w_k_t = w_k + (w_k - 1) * (dilation[1] - 1)
    ret = []
    for i in range(0, h_i - h_k_t + 1, stride[0]):
        for j in range(0, w_i - w_k_t + 1, stride[1]):
            m = inp[:, :, i:i + h_k_t:dilation[0], j:j + w_k_t:dilation[1]]
            ret.append(m.reshape(bs, 1, ch_i * h_k * w_k))
    return concat(ret, axis=1)
