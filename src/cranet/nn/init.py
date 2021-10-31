from ..autograd.tensor import Shapable
from typing import (
    Union,
)

import numpy as np


# TODO: impl initializers
# TODO: impl with tensor.zero_grad()

def uniform_(shape: Shapable, a: float = 0., b: float = 1.) -> np.ndarray:
    r"""Fills the input Tensor with values drawn from the uniform
    distribution.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    return np.random.uniform(a, b, shape)


def normal_(shape: Shapable, mean: float = 0., std: float = 1.) -> np.ndarray:
    r"""Fills the input Tensor with values drawn from the normal
    distribution.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    """
    return np.random.normal(mean, std, shape)


def constant_(shape: Shapable, val) -> np.ndarray:
    r"""Fills the input Tensor with the value :math:`\text{val}`.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
        val: the value to fill the tensor with

    """
    return np.full(shape, val)


def ones_(shape: Shapable) -> np.ndarray:
    r"""Fills the input Tensor with the scalar value `1`.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
    """
    return np.ones(shape)


def zeros_(shape: Shapable) -> np.ndarray:
    r"""Fills the input Tensor with the scalar value `0`.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
    """
    return np.zeros(shape)


def eye_(shape: Shapable) -> np.ndarray:
    r"""Fills the 2-dimensional input `Tensor` with the identity
    matrix. Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
    """
    return np.eye(*shape)


def calculate_gain(nonlinearity: str, param=None) -> Union[int, float]:
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_correct_fan(shape: Shapable, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(shape: Shapable):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = shape[0]
    num_output_fmaps = shape[1]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = shape[2:].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(shape: Shapable, gain: float = 1.) -> np.ndarray:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-a, a, shape)


def xavier_normal_(shape: Shapable, gain=1.) -> np.ndarray:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    return np.random.normal(0., std, shape)


def kaiming_uniform_(shape: Shapable, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if 0 in shape:
        raise ValueError("Initializing zero-element tensors is a no-op")

    fan = _calculate_correct_fan(shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return np.random.uniform(-bound, bound, shape)


def kaiming_normal_(shape: Shapable, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if 0 in shape:
        raise ValueError("Initializing zero-element tensors is a no-op")

    fan = _calculate_correct_fan(shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return np.random.normal(0., std, shape)
