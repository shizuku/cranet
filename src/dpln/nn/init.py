from typing import (
    Tuple,
    Union,
    List,
)

import numpy as np

Shapable = Union[Tuple, List]


# TODO: impl initializers

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
        shape:
    """
    return np.eye(*shape)


def dirac_(shape: Shapable, groups: int = 1) -> np.ndarray:
    r"""Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac
    delta function. Preserves the identi    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_outty of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        shape: tuple or list which determines the shape of initialization matrix
        groups (optional): number of groups in the conv layer (default: 1)
    """
    pass


def _calculate_fan_in_and_fan_out(shape: Shapable):
    # dimensions = tensor.dim()
    # if dimensions < 2:
    #     raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    #
    # num_input_fmaps = tensor.size(1)
    # num_output_fmaps = tensor.size(0)
    # receptive_field_size = 1
    # if tensor.dim() > 2:
    #     receptive_field_size = tensor[0][0].numel()
    # fan_in = num_input_fmaps * receptive_field_size
    # fan_out = num_output_fmaps * receptive_field_size
    #
    # return fan_in, fan_out
    pass


def xavier_uniform_(shape: Shapable, gain: float = 1.) -> np.ndarray:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
        gain: an optional scaling factor
    """
    return


def xavier_normal_(shape: Shapable, gain=1.) -> np.ndarray:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        shape: tuple or list which determines the shape of initialization matrix
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    return


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """


def sparse_(tensor, sparsity, std=0.01):
    r"""Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
