from __future__ import annotations

from .tensor import Tensor, Dependency

import numpy as np


def log(x: Tensor) -> Tensor:
    data = np.log(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 / x.data)

        dependencies.append(Dependency(x, grad_fn))

    return Tensor(data, requires_grad, dependencies)


def exp(x: Tensor) -> Tensor:
    data = np.exp(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data

        dependencies.append(Dependency(x, grad_fn))

    return Tensor(data, requires_grad, dependencies)


def relu(x: Tensor) -> Tensor:
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
    data = 1 / (1 + np.exp(-x.data))
    requires_grad = x.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


def softmax(x: Tensor, axis=-1) -> Tensor:
    e = np.exp(x.data - np.max(x.data))
    data = e / np.sum(e, axis=axis, keepdims=True)  # (bs, n)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            s = data[:, :, np.newaxis]  # (bs, n, 1)
            a = np.array([np.diagflat(s[i, :, :]) for i in range(s.shape[0])])  # (bs, n, n)
            b = np.matmul(s, s.transpose((0, 2, 1)))  # (bs, n, n)
            c = grad[:, np.newaxis, :]
            d = np.matmul(c, (a - b))
            return d.squeeze()

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)


def tanh(x: Tensor) -> Tensor:
    data = np.tanh(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        dependencies = [Dependency(x, grad_fn)]
    else:
        dependencies = []

    return Tensor(data, requires_grad, dependencies)
