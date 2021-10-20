from ..autograd import Tensor, concat

from typing import (
    Union, Tuple, List,
)


def im2col2d(inp: Tensor, kernel_shape: Union[Tuple, List], stride: Tuple):
    """

    :param inp: Tensor (bs, ch_i, h_i, w_i)
    :param kernel_shape: (ch_o, ch_i, h_k, w_k)
    :param stride: Tuple[int, int]
    :return: Tensor (bs, *, i_ch*k_w*k_h)
    """
    bs, _, h_i, w_i = inp.shape
    _, ch_i, h_k, w_k = kernel_shape
    ret = []
    for i in range(0, h_i - h_k + 1, stride[0]):
        for j in range(0, w_i - w_k + 1, stride[1]):
            ret.append(inp[:, :, i:i + h_k, j:j + w_k].reshape(bs, 1, ch_i * h_k * w_k))
    return concat(ret, axis=1)
