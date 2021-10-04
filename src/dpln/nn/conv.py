import numpy as np

from .optimizer import Optimizer
from .module import Module


class Conv2D(Module):
    def __init__(self, filters: int, kernal_size: int, strides=(1, 1), padding=0, bias=True):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return super().forward(x)

    def backward(self, dl: np.ndarray) -> np.ndarray:
        return super().backward(dl)

    def update(self, optimizer: Optimizer):
        return super().update(optimizer)
