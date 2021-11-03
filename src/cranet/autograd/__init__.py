from .tensor import (
    Tensor,
    tensor, as_tensor, stack,
    is_grad_enabled, set_grad_enabled,
    empty, zeros, zeros_like, ones, ones_like,
    add, sub, neg, mul, truediv, matmul, power, dot,
    sum, mean, var,
    transpose, permute, concat, reshape, flatten, pad
)
from .function import (
    abs, max, min, log, exp, relu, sigmoid, softmax, tanh,
)
from .random import (
    normal, normal_like,
    uniform, uniform_like,
)
from .grad_mode import (
    no_grad
)
import numpy as np

bool = np.bool_
uint8 = np.uint8
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
long = np.int32
float16 = np.float16
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128
