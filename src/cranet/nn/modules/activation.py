from cranet import Tensor

from .module import Module
from .. import functional as F


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, dim=self.dim)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)
