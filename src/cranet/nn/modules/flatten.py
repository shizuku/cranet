from cranet import Tensor

from .module import Module
from .. import functional as F


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return F.flatten(x, self.start_dim, self.end_dim)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"
