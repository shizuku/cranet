import numpy as np


class Loss:
    def forward(self, a: np.ndarray, y: np.ndarray):
        pass

    def backward(self, a: np.ndarray, y: np.ndarray):
        pass

    def __call__(self, a, y):
        return self.forward(a, y)


class MSELoss(Loss):
    def forward(self, a: np.ndarray, y: np.ndarray):
        return np.average(np.sum((a - y) ** 2, axis=-1) / 2)

    def backward(self, a: np.ndarray, y: np.ndarray):
        return a - y


class CrossEntropyLoss(Loss):
    def forward(self, p: np.ndarray, y: np.ndarray):
        return -np.average(np.sum(y * np.log(p+1e-10), axis=-1))

    def backward(self, a: np.ndarray, y: np.ndarray):
        return a - y
