from __future__ import annotations

from typing import (
    List,
    NamedTuple,
    Callable,
    Optional,
    Union
)

import numpy as np


class Dependency(NamedTuple):
    tensor: Tensor
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False, dependencies: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.dependencies = dependencies or []
        self.shape = self.data.shape
        self.grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: Optional[Tensor] = None) -> None:
        assert self.requires_grad, "called backward on a non-requires-grad tensor"
        assert self.grad is not None, "must call zero_grad before backward"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-zero-tensor")
        self.grad.data += grad.data

        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> Tensor:
        return sum(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.requires_grad}, dependencies={self.dependencies})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return (self.data == other.data).all()


def sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and return the sum of its components
    """
    data = t.data.sum()
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)

        dependencies.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, dependencies)


def zeros(shape, dtype=None, order='C', requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape=shape, dtype=dtype, order=order), requires_grad=requires_grad)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None, requires_grad=False) -> Tensor:
    return Tensor(np.zeros_like(a=a, dtype=dtype, order=order, subok=subok, shape=shape), requires_grad=requires_grad)


def ones(shape, dtype=None, order='C', *, like=None, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape=shape, dtype=dtype, order=order, like=like), requires_grad=requires_grad)


def ones_like(a, dtype=None, order='K', subok=True, shape=None, requires_grad=False) -> Tensor:
    return Tensor(np.ones_like(a=a, dtype=dtype, order=order, subok=subok, shape=shape), requires_grad=requires_grad)


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.add(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data - t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return -grad

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


def neg(t: Tensor) -> Tensor:
    data = np.negative(t.data)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        dependencies.append(Dependency(t, lambda x: -x))

    return Tensor(data, requires_grad, dependencies)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


def div(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.divide(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad / t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = - grad * t1.data / t2.data ** 2
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)
