import numpy as np

from .optimizer import Optimizer
from .module import Module

from ..function import softmax, sigmoid


class ReLU(Module):
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


class Sigmoid(Module):
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


class Softmax(Module):
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
