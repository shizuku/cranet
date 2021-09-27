import numpy as np

from .optimizer import Optimizer
from .module import Module

from ..util import random_initializer


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, use_bias=True, initializer=random_initializer):
        super().__init__()
        self.use_bias = use_bias
        self.W_shape = [in_features, out_features]
        self.W = initializer(self.W_shape)  # (in, out)
        self.dW = None  # (in, out)
        if use_bias:
            self.b_shape = [out_features]
            self.b = initializer(self.b_shape)  # (out)
            self.db = None  # (out)
        else:
            self.b_shape = None
            self.b = None
            self.db = None
        self.x = None  # (bs, in)
        self.z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Linear forward

        :param x: shape(batch_size, in_features)
        :return: shape(batch_size, out_features)
        """
        self.x = x  # (bs, in)
        self.z = np.dot(self.x, self.W)  # (bs, out)
        if self.use_bias:
            self.z = np.add(self.z, self.b)
        return self.z

    def backward(self, dz: np.ndarray):
        """
        Linear backward

        :param dz: shape(batch_size, out_features)
        :return: shape(batch_size, in_features)
        """
        dx = np.dot(dz, self.W.transpose())
        self.dW = np.dot(self.x.transpose(), dz)
        if self.use_bias:
            self.db = np.sum(dz, axis=0)  # (bs, out)
        return dx

    def update(self, optimizer: Optimizer):
        self.W = optimizer.update(self.W, self.dW)
        if self.use_bias:
            self.b = optimizer.update(self.b, self.db)
