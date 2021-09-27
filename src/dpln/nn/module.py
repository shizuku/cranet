import numpy as np

from .optimizer import Optimizer


class Module:
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
