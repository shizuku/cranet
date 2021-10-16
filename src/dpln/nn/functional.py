from ..autograd.tensor import Tensor, Dependency

import numpy as np

from typing import (
    Optional,
)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = Ax + b`.
    """
    output = x @ weight
    if bias is None:
        output += bias
    ret = output
    return ret


def relu(x: Tensor) -> Tensor:
    # TODO: test
    data = np.maximum(x.data, 0)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (np.where(x.data.any() > 0, 1, 0))

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


def sigmoid(x: Tensor) -> Tensor:
    # TODO: test
    data = 1 / (1 + np.exp(-x))
    requires_grad = x.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


def softmax(x: Tensor, axis=-1) -> Tensor:
    # TODO: test
    e = np.exp(x.data - np.max(x))
    data = e / np.sum(e, axis=axis, keepdims=True)  # (bs, n)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            s = data[:, :, np.newaxis]  # (bs, n, 1)
            a = np.array([np.diagflat(s[i, :, :]) for i in range(s.shape[0])])  # (bs, n, n)
            b = np.matmul(s, s.transpose((0, 2, 1)))  # (bs, n, n)
            return grad * (a - b)

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


def tanh(x: Tensor) -> Tensor:
    # TODO: test
    data = np.tanh(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


def mse_loss(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    # TODO: test
    L = x - y

    if reduction == 'mean':
        return L.mean()
    elif reduction == 'sum':
        return L.sum()
    elif reduction is None:
        return L
    else:
        raise ValueError("reduction must be 'mean', 'sum', or None")


def cross_entropy(x: Tensor, y: Tensor, weight=None, reduction: str = 'mean') -> Tensor:
    # TODO: test
    pass


def l1_loss(x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pass
