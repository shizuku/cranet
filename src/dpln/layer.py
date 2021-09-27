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

    def backward(self, dl: np.ndarray) -> np.ndarray:
        return dl

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


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.mask = (z <= 0)
        a = z.copy()
        a[self.mask] = 0
        return a

    def backward(self, da: np.ndarray) -> np.ndarray:
        da[self.mask] = 0
        dz = da
        return dz

    def update(self, optimizer: Optimizer):
        pass


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.a = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid forward

        :param z: shape(batch_size, in)
        :return: shape(batch_size, in)
        """
        self.a = sigmoid(z)
        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """
        Sigmoid backward

        :param da: shape(batch_size, in)
        :return: shape(batch_size, in)
        """
        df = (1.0 - self.a) * self.a
        dz = da * df
        return dz

    def update(self, optimizer: Optimizer):
        pass


class Softmax(Layer):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.z = None
        self.a = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax forward

        :param z: shape(batch_size, in)
        :return: shape(batch_size, in)
        """
        self.z = z
        self.a = softmax(z, axis=self.axis)
        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """
        softmax backward

        :param da: shape(batch_size, in)
        :return: shape(batch_size, in)
        """
        def fn(z, daa):
            def df(Q):
                x = softmax(Q)
                s = x.reshape(-1, 1)
                return (np.diagflat(s) - np.dot(s, s.T))
            # z: (10)
            # dfz: (10, 10)
            # da: (10)
            dfz = df(z)
            return np.dot(dfz, daa)

        dz = []
        for b in range(da.shape[0]):
            dz.append(fn(self.z[b, :], da[b, :]))
        dz = np.array(dz)
        return dz

    def update(self, optimizer: Optimizer):
        pass
