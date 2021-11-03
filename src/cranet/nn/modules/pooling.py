from cranet import Tensor

from .module import Module
from .. import functional as F

from typing import (
    Tuple,
    List,
    Union,
)


class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[Tuple, List, int],
                 stride: Union[Tuple, List, int] = None,
                 padding: Union[Tuple, List, int] = 0,
                 dilation: Union[Tuple, List, int] = 1,
                 padding_mode: str = 'zeros'):
        super().__init__()

        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        elif type(kernel_size) in [tuple, list]:
            self.kernel_size = kernel_size
        else:
            raise ValueError("value of `kernel_size` must be int, tuple or list")

        if stride is None:
            self.stride = self.kernel_size
        elif type(stride) == int:
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

        self.padding_mode = padding_mode

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.padding_mode)

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
