from .module import Module
from .. import functional as F
from src.dpln.autograd.tensor import Tensor

from typing import (
    Optional,
)

class _Loss(Module):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.register_buffer('weight', weight)


class L1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)

class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)

class CrossEntropyLoss(_WeightedLoss):
    ...