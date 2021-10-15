from __future__ import annotations

import numpy as np

from .tensor import Tensor, Dependency


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
