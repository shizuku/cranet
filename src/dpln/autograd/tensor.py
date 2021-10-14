from __future__ import annotations

from typing import (
    List,
    Tuple,
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


Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> Tensor:
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


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

    def T(self):
        pass

    def sum(self) -> Tensor:
        return _sum(self)

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        return _transpose(self, dim1, dim2)

    def permute(self, axes: Union[List, Tuple]) -> Tensor:
        return _permute(self, axes)

    def __add__(self, other) -> Tensor:
        """gets called if I do t + other"""
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> Tensor:
        """gets called if I do other + t"""
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> Tensor:
        """when we do t += other"""
        self.data = self.data + ensure_tensor(other).data
        return self

    def __sub__(self, other) -> Tensor:
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> Tensor:
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other) -> Tensor:
        """when we do t -= other"""
        self.data = self.data - ensure_tensor(other).data
        return self

    def __neg__(self) -> Tensor:
        return _neg(self)

    def __mul__(self, other) -> Tensor:
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> Tensor:
        return _mul(ensure_tensor(other), self)

    def __imul__(self, other) -> Tensor:
        """when we do t *= other"""
        self.data = self.data * ensure_tensor(other).data
        return self

    def __truediv__(self, other) -> Tensor:
        return _truediv(self, ensure_tensor(other))

    def __rtruediv__(self, other) -> Tensor:
        return _truediv(ensure_tensor(other), self)

    def __itruediv__(self, other) -> Tensor:
        self.data = self.data / ensure_tensor(other).data
        return self

    def __matmul__(self, other) -> Tensor:
        return _matmul(self, other)

    def __getitem__(self, idxs):
        return _slice(self, idxs)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.requires_grad}, dependencies={self.dependencies})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return (self.data == other.data).all()


def zeros(shape, dtype=None, order='C', requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape=shape, dtype=dtype, order=order), requires_grad=requires_grad)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None, requires_grad=False) -> Tensor:
    return Tensor(np.zeros_like(a=a, dtype=dtype, order=order, subok=subok, shape=shape), requires_grad=requires_grad)


def ones(shape, dtype=None, order='C', *, like=None, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape=shape, dtype=dtype, order=order, like=like), requires_grad=requires_grad)


def ones_like(a: Tensor, dtype=None, requires_grad=False) -> Tensor:
    if dtype is None:
        dtype = a.data.dtype
    return Tensor(np.ones_like(a=a, dtype=dtype, shape=a.shape), requires_grad=requires_grad)


def _sum(t: Tensor) -> Tensor:
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


def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        dependencies.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, dependencies)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
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


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
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


def _neg(t: Tensor) -> Tensor:
    data = np.negative(t.data)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        dependencies.append(Dependency(t, lambda x: -x))

    return Tensor(data, requires_grad, dependencies)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
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


def _truediv(t1: Tensor, t2: Tensor) -> Tensor:
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


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ np.swapaxes(t2.data, -1, -2)

        dependencies.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return np.swapaxes(t1.data, -1, -2) @ grad

        dependencies.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, dependencies)


def _dot(t1: Tensor, t2: Tensor) -> Tensor:
    pass


def _transpose(a: Tensor, dim1: int, dim2: int) -> Tensor:
    data = np.swapaxes(a.data, dim1, dim2)
    requires_grad = a.requires_grad
    dependencies: List[Dependency] = []

    if a.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return np.swapaxes(grad, dim2, dim1)

        dependencies.append(Dependency(a, grad_fn))

    return Tensor(data, requires_grad, dependencies)


def _permute(a: Tensor, axes: Union[List, Tuple]) -> Tensor:
    data = a.data.transpose(axes)
    requires_grad = a.requires_grad
    dependencies: List[Dependency] = []

    if a.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.transpose(axes)

        dependencies.append(Dependency(a, grad_fn))

    return Tensor(data, requires_grad, dependencies)
