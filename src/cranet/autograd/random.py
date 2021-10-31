from .tensor import Tensor

import numpy as np


def normal(shape, loc=0.0, scale=1.0, requires_grad=False) -> Tensor:
    return Tensor(np.random.normal(loc=loc, scale=scale, size=shape), requires_grad=requires_grad)


def normal_like(a, loc=0.0, scale=1.0, requires_grad=False) -> Tensor:
    return Tensor(np.random.normal(loc=loc, scale=scale, size=a.shape), requires_grad=requires_grad)


def uniform(shape, low=0.0, high=1.0, requires_grad=False) -> Tensor:
    return Tensor(np.random.uniform(low=low, high=high, size=shape), requires_grad=requires_grad)


def uniform_like(a, low=0.0, high=1.0, requires_grad=False) -> Tensor:
    return Tensor(np.random.uniform(low=low, high=high, size=a.shape), requires_grad=requires_grad)
