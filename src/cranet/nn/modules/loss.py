from abc import ABC

from .module import Module
from .. import functional as F
from cranet import Tensor

from typing import (
    Optional,
)


class _Loss(Module, ABC):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def __call__(self, inp: Tensor, tar: Tensor):
        return self.forward(inp, tar)


class _WeightedLoss(_Loss, ABC):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.register_buffer('weight', weight)


class L1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__(reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(x, y, reduction=self.reduction)


class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.mse_loss(x, y, reduction=self.reduction)


class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(weight, reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(x, y, weight=self.weight, reduction=self.reduction)


class NLLLoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(weight, reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(x, y, weight=self.weight, reduction=self.reduction)
