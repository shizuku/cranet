import numpy as np

from .function import softmax, softmax_derivative, sigmoid
from .optimizer import Optimizer
from .util import random_initializer


class Layer:
    def __init__(self):
        self.name = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout

    def update(self, optimizer: Optimizer):
        pass


class Linear(Layer):
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
        forward of linear: $z = Wx+b$

        :param x: shape(batch_size, in_features)
        :return: shape(batch_size, out_features)
        """
        self.x = x  # (bs, in)
        self.z = np.dot(self.x, self.W)  # (bs, out)
        if self.use_bias:
            self.z = np.add(self.z, self.b)
        return self.z

    def backward(self, dout: np.ndarray):
        """
        Linear backward

        :param dout:  shape(bs, out_features)
        :return: (bs, in_features)
        """
        dx = np.dot(dout, self.W.transpose())
        self.dW = np.dot(self.x.transpose(), dout)
        if self.use_bias:
            self.db = np.sum(dout, axis=0)  # (bs, out)
        return dx

    def update(self, optimizer: Optimizer):
        self.W = optimizer.update(self.W, self.dW)
        if self.use_bias:
            self.b = optimizer.update(self.b, self.db)


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout
        return dx

    def update(self, optimizer: Optimizer):
        pass


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.a = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        sigmoid forward

        :param z: (bs, in)
        :return: (bs, in)
        """
        self.a = sigmoid(z)
        return self.a

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        sigmoid backward

        :param dout: (bs, in)
        :return: (bs, in)
        """
        df = (1.0 - self.a) * self.a
        dx = dout * df
        return dx

    def update(self, optimizer: Optimizer):
        pass


class Softmax(Layer):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.z = None
        self.a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        softmax forward

        :param x: shape: (batch_size, 10)
        :return: shape: (batch_size, 10)
        """
        self.z = x
        self.a = softmax(x, axis=self.axis)
        return self.a

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        softmax backward

        :param dout: shape(batch_size, 10)
        :return: shape(batch_size, 10)
        """
        df = softmax_derivative(self.z)  # (bs, 10, 10)
        dx = np.matmul(df, dout[:, :, np.newaxis])  # (bs, 10)
        dx = dx[:, :, 0]
        return dx

    def update(self, optimizer: Optimizer):
        pass
