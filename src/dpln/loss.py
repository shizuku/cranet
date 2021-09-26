import numpy as np


class Loss:
    def forward(self, a: np.ndarray, y: np.ndarray):
        pass

    def backward(self, a: np.ndarray, y: np.ndarray):
        pass

    def __call__(self, a, y):
        return self.forward(a, y)


class MSE(Loss):
    def forward(self, a: np.ndarray, y: np.ndarray):
        return np.average(np.sum((a - y) ** 2, axis=-1) / 2)

    def backward(self, a: np.ndarray, y: np.ndarray):
        return a - y


class CrossEntropy(Loss):
    def forward(self, p: np.ndarray, y: np.ndarray):
        return np.tensordot(y, np.log(p), axes=[[1], [1]])

    def backward(self, p: np.ndarray, y: np.ndarray):
        pass
