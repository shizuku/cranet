from __future__ import annotations

from .tensor import Tensor, Dependency

import numpy as np


def abs(x: Tensor) -> Tensor:
    data = np.absolute(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * np.where(x.data > 0, 1, -1)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "abs"}))

    return Tensor(data, requires_grad, dependencies)


def max(x: Tensor, axis=None, keepdims=False) -> Tensor:
    # TODO: fix epsilon
    data = np.amax(x.data, axis=axis, keepdims=keepdims)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            if axis is None:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmax()
                idx = np.unravel_index(x_am, x.shape)
                ret[idx] = grad
                return ret
            else:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmax(axis=axis)
                x_am = np.expand_dims(x_am, axis=axis)
                grad_r = grad if keepdims else np.expand_dims(grad, axis=axis)
                np.put_along_axis(ret, x_am, grad_r, axis=axis)
                return ret

        dependencies.append(Dependency(x, grad_fn, meta={"name": "max"}))

    return Tensor(data, requires_grad, dependencies)


def min(x: Tensor, axis=None, keepdims=False) -> Tensor:
    # TODO: fix epsilon
    data = np.amin(x.data, axis=axis, keepdims=keepdims)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            if axis is None:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmin()
                idx = np.unravel_index(x_am, x.shape)
                ret[idx] = grad
                return ret
            else:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmin(axis=axis)
                x_am = np.expand_dims(x_am, axis=axis)
                grad_r = grad if keepdims else np.expand_dims(grad, axis=axis)
                np.put_along_axis(ret, x_am, grad_r, axis=axis)
                return ret

        dependencies.append(Dependency(x, grad_fn, meta={"name": "max"}))

    return Tensor(data, requires_grad, dependencies)


def log(x: Tensor) -> Tensor:
    data = np.log(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * (1 / x.data)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "log"}))

    return Tensor(data, requires_grad, dependencies)


def exp(x: Tensor) -> Tensor:
    data = np.exp(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * data

        dependencies.append(Dependency(x, grad_fn, meta={"name": "exp"}))

    return Tensor(data, requires_grad, dependencies)


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if training:
        mask = np.random.rand(*x.shape) > p
        data = x.data * mask * (1 / (1 - p))
        requires_grad = x.requires_grad
        dependencies = []

        if x.requires_grad:
            def grad_fn(grad: np.ndarray, _) -> np.ndarray:
                return mask * grad / (1 - p)

            dependencies.append(Dependency(x, grad_fn, meta={"name": "dropout"}))

        return Tensor(data, requires_grad, dependencies)
    else:
        return x


def relu(x: Tensor) -> Tensor:
    data = np.maximum(x.data, 0)
    requires_grad = x.requires_grad
    dependencies = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * np.where(x.data > 0, 1, 0)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "relu"}))

    return Tensor(data, requires_grad, dependencies)


def sigmoid(x: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-x.data))
    requires_grad = x.requires_grad
    dependencies = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * data * (1 - data)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "sigmoid"}))

    return Tensor(data, requires_grad, dependencies)


def softmax(x: Tensor, axis=-1) -> Tensor:
    e = np.exp(x.data - np.max(x.data))
    data = e / np.sum(e, axis=axis, keepdims=True)  # (bs, n)
    requires_grad = x.requires_grad
    dependencies = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            s = data[:, :, np.newaxis]  # (bs, n, 1)
            a = np.array([np.diagflat(s[i, :, :]) for i in range(s.shape[0])])  # (bs, n, n)
            b = np.matmul(s, s.transpose((0, 2, 1)))  # (bs, n, n)
            c = grad[:, np.newaxis, :]
            d = np.matmul(c, (a - b))
            return d.squeeze()

        dependencies.append(Dependency(x, grad_fn, meta={"name": "softmax"}))

    return Tensor(data, requires_grad, dependencies)


def tanh(x: Tensor) -> Tensor:
    data = np.tanh(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * (1 - data * data)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "tanh"}))

    return Tensor(data, requires_grad, dependencies)
