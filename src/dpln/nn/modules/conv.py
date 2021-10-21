from dpln import Tensor

from .module import Module
from .. import functional as F
from ..parameter import Parameter
from .. import init

import math
from typing import (
    Optional,
    Tuple,
    List,
    Union,
)


class Conv2d(Module):
    in_channels: int
    out_channels: int
    kernel_size: Union[Tuple, List, int]
    stride: Union[Tuple, List, int]
    padding: Union[Tuple, List, int, str]
    dilation: Union[Tuple, List, int]
    groups: int
    padding_mode: str
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[Tuple, List, int],
                 stride: Union[Tuple, List, int] = 1,
                 padding: Union[Tuple, List, int, str] = 0,
                 dilation: Union[Tuple, List, int] = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        """
            TODO: impl dilation, groups

            :param in_channels: str
            :param out_channels: str
            :param kernel_size: int | Tuple[int, int]
            :param stride: int | Tuple[int, int]
            :param padding: int | Tuple[int, int]
            :param dilation: int | Tuple[int, int]
            :param groups: int
            :param bias: bool
            :param padding_mode: str 'zeros'| 'reflect'| 'replicate'| 'circular'
            :return: Tensor shape(bs, ch_o, h_o, w_o)
            """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) in [tuple, list]:
            self.kernel_size = kernel_size
        else:
            raise ValueError("value of `kernel_size` must be int, tuple or list")

        if type(stride) == int:
            self.stride = (stride, stride)
        elif type(stride) in [tuple, list]:
            assert len(stride) == 2
            self.stride = stride
        else:
            raise ValueError("value of `stride` must be int, tuple or list")

        if type(padding) == int:
            self.padding = (padding, padding)
        elif type(padding) in [tuple, list]:
            self.padding = padding
        elif type(padding) == str:
            self.padding = padding
        else:
            raise ValueError("value of `padding` must be int, tuple, list or str")

        if type(dilation) == int:
            self.dilation = (dilation, dilation)
        elif type(dilation) in [tuple, list]:
            self.dilation = dilation
        else:
            raise ValueError("value of `dilation` must be int, tuple or list")

        self.groups = groups
        self.padding_mode = padding_mode
        self.reset_parameters(bias)

    def reset_parameters(self, use_bias: bool = True) -> None:
        """init(reset) parameters"""
        sqrt_k = math.sqrt(self.groups / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        weight_data = init.uniform_([self.out_channels, self.in_channels / self.groups, self.kernel_size[0], self.kernel_size[1]], -sqrt_k, sqrt_k)
        self.weight = Parameter(weight_data)
        if use_bias:
            bias_data = init.uniform_([self.out_channels], -sqrt_k, sqrt_k)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        padding = self.padding
        if type(self.padding) == str:
            padding = str2pad2d(self.padding, x.shape, self.weight.shape, self.stride)
        return F.conv2d(x, self.weight, self.bias, self.stride, padding,
                        self.dilation, self.groups, self.padding_mode)

    def __repr__(self) -> str:
        return f"Conv2d"


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
