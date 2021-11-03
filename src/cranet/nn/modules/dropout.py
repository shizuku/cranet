from cranet import Tensor

from .module import Module
from .. import functional as F


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
