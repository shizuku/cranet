from dpln import Tensor

from .module import Module
from .. import init
from .. import functional as F
from ..parameter import Parameter

from typing import (
    Optional,
)


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Softmax(Module):

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x)


class Tanh(Module):

    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)
