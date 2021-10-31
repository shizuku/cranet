from .function import (
    abs, max, min, log, exp, relu, sigmoid, softmax, tanh,
)
from .random import (
    normal, normal_like,
    uniform, uniform_like,
)
from .tensor import (
    Tensor,
    tensor, as_tensor, stack,
    zeros, zeros_like, ones, ones_like,
    add, sub, neg, mul, truediv, matmul, power, dot,
    sum, mean,
    transpose, permute, concat, reshape, flatten, pad
)
import numpy as np

uint8 = np.uint8
int32 = np.int32
int64 = np.int64
float32 = np.float32
float64 = np.float64
