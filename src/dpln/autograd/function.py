from __future__ import annotations

from .tensor import Tensor, Dependency
from . import utils as U

import numpy as np

import math
from typing import (
    Optional, Union, List, Tuple
)


def abs(x: Tensor) -> Tensor:
    data = np.absolute(x.data)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            return grad * np.where(x.data > 0, 1, -1)

        dependencies.append(Dependency(x, grad_fn, meta={"name": "abs"}))

    return Tensor(data, requires_grad, dependencies)


def max(x: Tensor, dim=None, keepdim=False) -> Tensor:
    # TODO: fix epsilon
    data = np.amax(x.data, axis=dim, keepdims=keepdim)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            if dim is None:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmax()
                idx = np.unravel_index(x_am, x.shape)
                ret[idx] = grad
                return ret
            else:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmax(axis=dim)
                x_am = np.expand_dims(x_am, axis=dim)
                grad_r = grad if keepdim else np.expand_dims(grad, axis=dim)
                np.put_along_axis(ret, x_am, grad_r, axis=dim)
                return ret

        dependencies.append(Dependency(x, grad_fn, meta={"name": "max"}))

    return Tensor(data, requires_grad, dependencies)


def min(x: Tensor, dim=None, keepdim=False) -> Tensor:
    # TODO: fix epsilon
    data = np.amin(x.data, axis=dim, keepdims=keepdim)
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            if dim is None:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmin()
                idx = np.unravel_index(x_am, x.shape)
                ret[idx] = grad
                return ret
            else:
                ret = np.zeros_like(x.data)
                x_am = x.data.argmin(axis=dim)
                x_am = np.expand_dims(x_am, axis=dim)
                grad_r = grad if keepdim else np.expand_dims(grad, axis=dim)
                np.put_along_axis(ret, x_am, grad_r, axis=dim)
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


def conv2d(x: Tensor, weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Union[Tuple, List, int] = 1,
           padding: Union[Tuple, List, int, str] = 0,
           dilation: Union[Tuple, List, int] = 1,
           groups: int = 1,
           padding_mode: str = 'zeros') -> Tensor:
    """

    :param x: (bs, ch_i, h_i, w_i)
    :param weight: (ch_o, ch_i_g, h_k, w_k)
    :param bias: (ch_o)
    :param stride: -> (int, int)
    :param padding: -> ((int, int), (int, int))
    :param dilation: -> (int, int)
    :param groups: int
    :param padding_mode: 'zeros'
    :return (bs, ch_o, h_o, w_o)
    """
    assert x.ndim == 4
    bs, ch_i, h_i, w_i = x.shape
    ch_o, ch_i_g, h_k, w_k = weight.shape
    assert bias is None or bias.ndim == 1

    # make sure stride is (int, int)
    stride_err_msg = "value of `stride` must be tuple of 2 or int"
    if type(stride) == int:
        stride = (stride, stride)
    elif type(stride) in [tuple, list]:
        assert len(stride) == 2, stride_err_msg
    else:
        raise ValueError(stride_err_msg)

    # make sure padding is ((int, int), (int,int))
    pad_err_msg = "value of `padding` must be tuple or list of 2, 2x2 or 4 or int"
    if type(padding) == int:
        padding = ((padding, padding), (padding, padding))
    elif type(padding) in [tuple, list]:
        if len(padding) == 2:
            if type(padding[0]) == int:
                padding = ((padding[0], padding[0]), padding[1])
            elif type(padding[0]) in [tuple, list]:
                assert len(padding[0]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
            if type(padding[1]) == int:
                padding = (padding[0], (padding[1], padding[1]))
            elif type(padding[1]) in [tuple, list]:
                assert len(padding[1]) == 2, pad_err_msg
            else:
                raise ValueError(pad_err_msg)
        elif len(padding) == 4:
            padding = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(pad_err_msg)
    elif type(padding) == str:
        padding = U.str2pad2d(padding, x.shape, weight.shape, stride)
    else:
        raise ValueError(pad_err_msg)

    # make sure dilation
    dia_err_msg = "value of `dilation` must be tuple of 2 or int"
    if type(dilation) == int:
        dilation = (dilation, dilation)
    elif type(dilation) in [tuple, int]:
        assert len(dilation) == 2, dia_err_msg
    else:
        raise ValueError(dia_err_msg)

    # make sure groups
    assert ch_i % groups == 0 and ch_o % groups == 0
    ch_o_g = ch_o // groups

    h_o = math.floor((h_i + padding[0][0] + padding[0][1] - dilation[0] * (h_k - 1) - 1)
                     / stride[0] + 1)
    w_o = math.floor((w_i + padding[1][0] + padding[1][1] - dilation[1] * (w_k - 1) - 1)
                     / stride[1] + 1)

    x_pad = U.padding2d(x.data, padding, padding_mode)
    col_x = U.im2col2d(x_pad, weight.shape, stride, dilation)
    col_x = col_x.reshape(bs, h_o * w_o, groups, ch_i_g * h_k * w_k)
    col_w = weight.reshape(groups, ch_o_g, ch_i_g * h_k * w_k)
    data = np.matmul(col_w.data, col_x, axes=[(1, 2), (3, 1), (2, 3)])
    data = data.reshape(bs, ch_o, h_o, w_o)
    if bias is not None:
        data + bias.reshape(1, ch_o, 1, 1)

    bias_requires_grad = bias is not None and bias.requires_grad
    requires_grad = x.requires_grad or weight.requires_grad or bias_requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn_x(grad: np.ndarray, _) -> np.ndarray:
            grad = np.transpose(grad, axes=(0, 2, 3, 1)).reshape(-1, ch_o)
            dcol = np.dot(grad, col_w.T)
            grad_x = U.col2img(dcol, x.shape, h_k, w_k, stride, padding)
            return grad_x

        dependencies.append(Dependency(x, grad_fn_x, meta={"name": "conv2d_x"}))

    if weight.requires_grad:
        def grad_fn_w(grad: np.ndarray, _) -> np.ndarray:
            grad = np.transpose(grad, axes=(0, 2, 3, 1)).reshape(-1, ch_o)
            grad_w = np.dot(col_x.T, grad) / bs
            grad_w = np.transpose(grad_w, axes=(1, 0)).reshape(ch_o, ch_i, h_k, w_k)
            return grad_w

        dependencies.append(Dependency(x, grad_fn_w, meta={"name": "conv2d_w"}))

    if bias_requires_grad:
        def grad_fn_b(grad: np.ndarray, _) -> np.ndarray:
            grad = np.transpose(grad, axes=(0, 2, 3, 1)).reshape(-1, ch_o)
            grad_b = np.sum(grad, axis=0, keepdims=True).T / bs
            return grad_b

        dependencies.append(Dependency(x, grad_fn_b, meta={"name": "conv2d_b"}))

    return Tensor(data, requires_grad, dependencies)


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


def softmax(x: Tensor, dim=-1) -> Tensor:
    e = np.exp(x.data - np.max(x.data))
    data = e / np.sum(e, axis=dim, keepdims=True)  # (bs, n)
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


def nll(x: Tensor, y: Tensor, w: Optional[Tensor] = None, dim=-1) -> Tensor:
    indices = np.expand_dims(y.data, axis=-1)
    data = np.take_along_axis(x.data, indices, axis=dim)
    if w:
        data = np.take_along_axis(w, indices, axis=0) * data
    data = data.squeeze()
    requires_grad = x.requires_grad
    dependencies = []

    if x.requires_grad:
        def grad_fn(grad: np.ndarray, _) -> np.ndarray:
            assert grad.shape == data.shape
            ret = np.zeros_like(x.data)
            np.put_along_axis(ret, indices, np.expand_dims(grad, -1), axis=dim)
            assert ret.shape == x.shape
            return ret

        dependencies.append(Dependency(x, grad_fn, meta={"name": "nll"}))

    return Tensor(data, requires_grad, dependencies)
