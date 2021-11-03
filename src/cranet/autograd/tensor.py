from __future__ import annotations

from .utils import invert_permutation

import numpy as np
from threading import local
from typing import (
    List,
    Tuple,
    Union,
    Sequence,
    Callable,
    Optional,
    NamedTuple,
    Dict,
)

tl = local()


class Dependency(NamedTuple):
    tensor: Tensor
    grad_fn: Callable[[np.ndarray, Optional[Dict]], np.ndarray]
    meta: Optional[Dict] = None


Shapable = Union[Tuple, List, int]
Arrayable = Union[float, list, np.ndarray]
Tensorable = Union['Tensor', np.ndarray, float, int, list]


def ensure_data(tensorable: Tensorable, dtype=None) -> np.ndarray:
    if isinstance(tensorable, Tensor):
        data = tensorable.data
        return data.astype(dtype)
    if isinstance(tensorable, np.ndarray):
        if dtype is None:
            return tensorable
        return tensorable.astype(dtype)
    else:
        return np.array(tensorable, dtype=dtype)


def ensure_tensor(tensorable: Tensorable, dtype=None) -> Tensor:
    if isinstance(tensorable, Tensor):
        if dtype is not None:
            tensorable.dtype = dtype
        return tensorable
    else:
        return Tensor(tensorable, dtype=dtype)


class Tensor:
    def __init__(self, data: Tensorable,
                 requires_grad: bool = False,
                 dependencies: List[Dependency] = None,
                 dtype=None) -> None:
        self._data = ensure_data(data, dtype=dtype)
        self.requires_grad = tl.is_grad_enabled and requires_grad
        self.dependencies = dependencies or []
        self.shape = self.data.shape
        self.grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new: np.ndarray) -> None:
        self._data = new
        # detach gradient if set new datasets
        self.grad = None

    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, new):
        self._data.dtype = new

    def astype(self, dtype) -> Tensor:
        new_data = self._data.copy()
        return Tensor(new_data, requires_grad=self.requires_grad,
                      dependencies=self.dependencies, dtype=dtype)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        if self.shape == ():
            return self.data
        elif self.shape == (1,):
            return self.data[0]
        else:
            raise RuntimeError('must be one element')

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: Optional[Tensor] = None) -> None:
        assert self.requires_grad, "called backward on a non-requires-grad tensor"
        assert self.grad is not None, "must call zero_grad before backward"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.)
            else:
                raise RuntimeError(
                    "grad must be specified for non-zero-tensor")
        self.grad.data += grad.data

        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad.data, dependency.meta)
            dependency.tensor.backward(Tensor(backward_grad))

    def dim(self) -> int:
        """Number of self.datasets dimensions"""
        return self._data.ndim

    def numel(self) -> int:
        """the product of the self.datasets’s dimensions."""
        return self._data.size

    def sum(self, dim: Optional[Shapable] = None, keepdim: bool = False) -> Tensor:
        return sum(self, dim, keepdim=keepdim)

    def mean(self, dim: Optional[Shapable] = None, keepdim: bool = False) -> Tensor:
        return mean(self, dim, keepdim=keepdim)

    def var(self, dim: Optional[Shapable] = None, keepdim: bool = False, unbiased: bool = True) -> Tensor:
        return var(self, dim, keepdim=keepdim, unbiased=unbiased)

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        return transpose(self, dim1, dim2)

    def permute(self, dims: Union[List, Tuple]) -> Tensor:
        return permute(self, dims)

    def reshape(self, *sp: int) -> Tensor:
        return reshape(self, sp)

    def flatten(self, start_dim=0, end_dim=-1):
        return flatten(self, start_dim, end_dim)

    def diag(self, diagonal: int = 0) -> Tensor:
        return diag(self, diagonal)

    def index_select(self, dim: int, index) -> Tensor:
        return index_select(self, dim, index)

    @property
    def T(self) -> Tensor:
        return permute(self, list(range(len(self.shape) - 1, -1, -1)))

    def detach(self) -> Tensor:
        new_data = self._data.copy()
        new_requires_grad = self.requires_grad
        return Tensor(new_data, new_requires_grad)

    def __add__(self, other) -> Tensor:
        """called if `self + other`"""
        return add(self, ensure_tensor(other))

    def __radd__(self, other) -> Tensor:
        """called if `other + self`"""
        return add(ensure_tensor(other), self)

    def __iadd__(self, other) -> Tensor:
        """called if `self += other`"""
        self.data = self.data + ensure_tensor(other).data
        return self

    def __sub__(self, other) -> Tensor:
        """called if `self - other`"""
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> Tensor:
        """called if `other - self`"""
        return sub(ensure_tensor(other), self)

    def __isub__(self, other) -> Tensor:
        """called if `self -= other`"""
        self.data = self.data - ensure_tensor(other).data
        return self

    def __neg__(self) -> Tensor:
        """called if `-self`"""
        return neg(self)

    def __mul__(self, other) -> Tensor:
        """called if `self * other`"""
        return mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> Tensor:
        """called if `other - self`"""
        return mul(ensure_tensor(other), self)

    def __imul__(self, other) -> Tensor:
        """called if `self *= other`"""
        self.data = self.data * ensure_tensor(other).data
        return self

    def __truediv__(self, other) -> Tensor:
        """called if `self / other`"""
        return truediv(self, ensure_tensor(other))

    def __rtruediv__(self, other) -> Tensor:
        """called if `other / self`"""
        return truediv(ensure_tensor(other), self)

    def __itruediv__(self, other) -> Tensor:
        """called if `self /= other`"""
        self.data = self.data / ensure_tensor(other).data
        return self

    def __matmul__(self, other) -> Tensor:
        """called if `self @ other`"""
        return matmul(self, ensure_tensor(other))

    def __rmatmul__(self, other) -> Tensor:
        """called if `other @ self`"""
        return matmul(ensure_tensor(other), self)

    def __imatmul__(self, other) -> Tensor:
        """called if `self @= other`"""
        self.data = self.data @ ensure_tensor(other).data
        return self

    def __pow__(self, other) -> Tensor:
        """called if `self ** other`"""
        return power(self, ensure_tensor(other))

    def __eq__(self, other: object) -> bool:
        """test if `self == other`"""
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.data.shape != other.data.shape:
            return False
        return (self.data == other.data).all()

    def __getitem__(self, idxs):
        return _slice(self, idxs)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.requires_grad})"

    def __hash__(self) -> int:
        return hash(str(self.data.data))


tl.is_grad_enabled = True


def is_grad_enabled() -> bool:
    return tl.is_grad_enabled


def set_grad_enabled(mode: bool = True):
    tl.is_grad_enabled = mode


def tensor(data: Arrayable, requires_grad: bool = False, dtype=None) -> Tensor:
    if type(data).__module__ == np.__name__:
        return Tensor(data.copy(), requires_grad=requires_grad, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def as_tensor(data: Arrayable, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    tensors = [unsqueeze(i, dim) for i in tensors]
    return concat(tensors, dim)


def empty(shape, dtype=None, requires_grad=False) -> Tensor:
    return Tensor(np.empty(shape=shape, dtype=dtype), requires_grad=requires_grad)


def zeros(shape, dtype=None, requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape=shape, dtype=dtype), requires_grad=requires_grad)


def zeros_like(a: Tensor, dtype=None, shape=None, requires_grad=False) -> Tensor:
    return Tensor(np.zeros_like(a=a, dtype=dtype, shape=shape), requires_grad=requires_grad)


def ones(shape, dtype=None, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape=shape, dtype=dtype), requires_grad=requires_grad)


def ones_like(a: Tensor, dtype=None, shape=None, requires_grad=False) -> Tensor:
    return Tensor(np.ones_like(a=a, dtype=dtype, shape=shape), requires_grad=requires_grad)


def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            ret = np.zeros_like(t.data)
            ret[idxs] = grad
            assert ret.shape == t.shape
            return ret

        dependencies.append(Dependency(t, grad_fn, meta={"name": "slice"}))

    return Tensor(data, requires_grad, dependencies)


def concat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    assert len(tensors) > 0
    i_dim = tensors[0].dim()
    data = np.concatenate([t.data for t in tensors], axis=dim)
    requires_grad = True in [t.requires_grad for t in tensors]
    dependencies: List[Dependency] = []

    a = 0
    b = 0
    for i, t in enumerate(tensors):
        b += tensors[i].shape[dim]
        if t.requires_grad:
            def grad_fn(grad: np.ndarray, meta) -> np.ndarray:
                idx = tuple([np.s_[meta["a"]:meta["b"]] if j == dim else np.s_[:] for j in range(i_dim)])
                return grad[idx]

            dependencies.append(Dependency(t, grad_fn, meta={"name": "concat", "id": i, "a": a, "b": b}))
        a += tensors[i].shape[dim]

    return Tensor(data, requires_grad, dependencies)


def unsqueeze(x: Tensor, dim: int) -> Tensor:
    data = np.expand_dims(x.data, axis=dim)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad.reshape(x.shape)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "unsqueeze"}))

    return Tensor(data, requires_grad, dependencies)


def reshape(t: Tensor, shape: Shapable) -> Tensor:
    data = t.data.reshape(shape)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if t.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad.reshape(t.shape)

        dependencies.append(Dependency(t, grad_fn, meta={"name": "reshape"}))

    return Tensor(data, requires_grad, dependencies)


def flatten(t: Tensor, start_dim=0, end_dim=-1) -> Tensor:
    if start_dim < 0:
        start_dim = t.data.ndim + start_dim
    if end_dim < 0:
        end_dim = t.data.ndim + end_dim
    sp = []
    for i in range(0, start_dim):
        sp.append(t.shape[i])
    p = 1
    for i in range(start_dim, end_dim + 1):
        p *= t.shape[i]
    sp.append(p)
    for i in range(end_dim + 1, t.data.ndim):
        sp.append(t.shape[i])
    data = t.data.reshape(sp)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if t.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad.reshape(t.shape)

        dependencies.append(Dependency(t, grad_fn, meta={"name": "flatten"}))

    return Tensor(data, requires_grad, dependencies)


def pad(t: Tensor, pad_width, mode='constant', **kwargs) -> Tensor:
    data = np.pad(t.data, pad_width, mode, **kwargs)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if t.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            idx = tuple([np.s_[pad_width[i][0]:grad.shape[i] - pad_width[i][1]] for i in range(grad.ndim)])
            return grad[idx]

        dependencies.append(Dependency(t, grad_fn, meta={"name": "pad"}))

    return Tensor(data, requires_grad, dependencies)


def index_select(t: Tensor, dim: int, index: List[int]) -> Tensor:
    # TODO: impl grad
    data = np.take(t.data, index, dim)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            pass

        dependencies.append(Dependency(t, grad_fn, meta={"name": "index_select"}))

    return Tensor(data, requires_grad, dependencies)


def diag(t: Tensor, diagonal: int = 0) -> Tensor:
    # TODO: impl grad
    data = np.diag(t.data, diagonal)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            pass

        dependencies.append(Dependency(t, grad_fn, meta={"name": "diag"}))

    return Tensor(data, requires_grad, dependencies)


def sum(x: Tensor, dim: Optional[Shapable] = None,
        keepdim: bool = False) -> Tensor:
    """
    Takes a tensor and return the sum of its components
    """
    # TODO: fix epsilon
    data = x.data.sum(axis=dim, keepdims=keepdim)
    requires_grad = x.requires_grad
    dependencies: List[Dependency] = []

    if x.requires_grad:
        if dim is None or keepdim:
            def grad_fn(grad: np.ndarray, _) -> np.ndarray:
                return grad * np.ones_like(x.data)
        elif type(dim) is int:
            def grad_fn(grad: np.ndarray, _) -> np.ndarray:
                grad = np.expand_dims(grad, axis=dim)
                return grad * np.ones_like(x.data)
        else:
            def grad_fn(grad: np.ndarray, _) -> np.ndarray:
                for i in sorted(dim):
                    grad = np.expand_dims(grad, axis=i)
                return grad * np.ones_like(x.data)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "sum"}))

    return Tensor(data, requires_grad, dependencies)


def mean(t: Tensor, dim: Optional[Shapable] = None,
         keepdim: bool = False) -> Tensor:
    data = t.sum(dim=dim, keepdim=keepdim)
    if dim is None:
        numel = t.numel()
    else:
        total_numel = t.numel()
        numel = total_numel // data.numel()
    return data / numel


def var(x: Tensor, dim: Optional[Shapable] = None,
        unbiased: bool = True, keepdim: bool = False) -> Tensor:
    z = (x - x.mean(dim=dim, keepdim=True)) ** 2
    s = z.sum(dim=dim, keepdim=keepdim)
    if dim is None:
        numel = z.numel()
    else:
        total_numel = z.numel()
        numel = total_numel // s.numel()
    if unbiased:
        numel -= 1
    return s / numel


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.add(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray, _) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1, meta={"name": "add_lhs"}))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray, _) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t2, grad_fn2, meta={"name": "add_rhs"}))

    return Tensor(data, requires_grad, dependencies)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data - t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray, _) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1, meta={"name": "sub_lhs"}))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray, _) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return -grad

        dependencies.append(Dependency(t2, grad_fn2, meta={"name": "sub_rhs"}))

    return Tensor(data, requires_grad, dependencies)


def neg(t: Tensor) -> Tensor:
    data = np.negative(t.data)
    requires_grad = t.requires_grad
    dependencies: List[Dependency] = []

    if requires_grad:
        dependencies.append(Dependency(t, lambda x, _: -x, meta={"name": "neg"}))

    return Tensor(data, requires_grad, dependencies)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray, _) -> np.ndarray:
            grad = grad * t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1, meta={"name": "mul_lhs"}))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray, _) -> np.ndarray:
            grad = grad * t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t2, grad_fn2, meta={"name": "mul_rhs"}))

    return Tensor(data, requires_grad, dependencies)


def truediv(t1: Tensor, t2: Tensor) -> Tensor:
    # TODO: fix epsilon
    data = np.divide(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray, _) -> np.ndarray:
            grad = np.divide(grad, t2.data)
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t1, grad_fn1, meta={"name": "truediv_lhs"}))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray, _) -> np.ndarray:
            grad = - (grad / t2.data) * (t1.data / t2.data)
            # grad = np.negative(np.divide(np.multiply(grad, t1.datasets), t2.datasets * t2.datasets))
            # grad = np.negative(np.divide(np.multiply(grad, t1.datasets), np.power(t2.datasets, 2)))
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(t2, grad_fn2, meta={"name": "truediv_rhs"}))

    return Tensor(data, requires_grad, dependencies)


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = np.matmul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray, _) -> np.ndarray:
            data_t1 = t1.data
            data_t2 = t2.data
            if t1.dim() == 1:
                data_t1 = np.expand_dims(data_t1, 0)
                grad = np.expand_dims(grad, -2)
            if t2.dim() == 1:
                data_t2 = np.expand_dims(data_t2, -1)
                grad = np.expand_dims(grad, -1)
            data_t2 = np.swapaxes(data_t2, -1, -2)
            grad_t1 = grad @ data_t2
            ndims_added = grad_t1.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad_t1 = grad_t1.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad_t1 = grad_t1.sum(axis=i, keepdims=True)
            assert grad_t1.shape == t1.shape
            return grad_t1

        dependencies.append(Dependency(t1, grad_fn1, meta={"name": "matmul_lhs"}))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray, _) -> np.ndarray:
            data_t1 = t1.data
            data_t2 = t2.data
            if t1.dim() == 1:
                data_t1 = np.expand_dims(data_t1, 0)
                grad = np.expand_dims(grad, -2)
            if t2.dim() == 1:
                data_t2 = np.expand_dims(data_t2, -1)
                grad = np.expand_dims(grad, -1)
            data_t1 = np.swapaxes(data_t1, -1, -2)
            grad_t2 = data_t1 @ grad
            if t2.dim() == 1:
                grad_t2 = np.squeeze(grad_t2, axis=-1)
            ndims_added = grad_t2.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad_t2 = grad_t2.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad_t2 = grad_t2.sum(axis=i, keepdims=True)
            assert grad_t2.shape == t2.shape
            return grad_t2

        dependencies.append(Dependency(t2, grad_fn2, meta={"name": "matmul_rhs"}))

    return Tensor(data, requires_grad, dependencies)


def power(x: Tensor, e: Tensor) -> Tensor:
    # TODO: fix epsilon
    data = np.power(x.data, e.data)
    requires_grad = x.requires_grad or e.requires_grad
    dependencies: List[Dependency] = []

    if x.requires_grad:
        def grad_fn1(grad: np.ndarray, _) -> np.ndarray:
            grad = grad * e.data * np.power(x.data, e.data - 1)
            ndims_added = grad.ndim - x.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(x.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(x, grad_fn1, meta={"name": "power_lhs"}))

    if e.requires_grad:
        def grad_fn2(grad: np.ndarray, _) -> np.ndarray:
            grad = grad * data * np.log(x.data)
            ndims_added = grad.ndim - e.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(e.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        dependencies.append(Dependency(e, grad_fn2, meta={"name": "power_rhs"}))

    return Tensor(data, requires_grad, dependencies)


def dot(t1: Tensor, t2: Tensor) -> Tensor:
    # TODO: impl, test
    assert t1.dim() <= 1 and t2.dim() <= 1
    data = np.dot(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    dependencies: List[Dependency] = []

    return Tensor(data, requires_grad, dependencies)


def transpose(a: Tensor, dim1: int, dim2: int) -> Tensor:
    data = np.swapaxes(a.data, dim1, dim2)
    requires_grad = a.requires_grad
    dependencies: List[Dependency] = []

    if a.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return np.swapaxes(grad, dim2, dim1)

        dependencies.append(Dependency(a, grad_fn, meta={"name": "transpose"}))

    return Tensor(data, requires_grad, dependencies)


def permute(a: Tensor, dims: Union[List, Tuple]) -> Tensor:
    data = a.data.transpose(dims)
    requires_grad = a.requires_grad
    dependencies: List[Dependency] = []

    if a.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            axes_t = invert_permutation(dims)
            return grad.transpose(axes_t)

        dependencies.append(Dependency(a, grad_fn, meta={"name": "permute"}))

    return Tensor(data, requires_grad, dependencies)
