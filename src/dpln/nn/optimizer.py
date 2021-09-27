import numpy as np


class Optimizer:
    def update(self, param: np.ndarray, grad: np.ndarray):
        pass


class SGD(Optimizer):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def update(self, params: np.ndarray, grads: np.ndarray):
        return params - self.alpha * grads
