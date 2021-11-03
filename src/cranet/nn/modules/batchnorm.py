import cranet
from cranet import Tensor

from .module import Module
from ..parameter import Parameter
from .. import functional as F

import numpy as np
from typing import Optional


class _NormBase(Module):
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool

    def __init__(self, num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(cranet.empty(num_features))
            self.bias = Parameter(cranet.empty(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', cranet.zeros(num_features))
            self.register_buffer('running_var', cranet.ones(num_features))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 cranet.tensor(0, dtype=cranet.long))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.register_buffer('running_mean', cranet.zeros(self.num_features))
            self.register_buffer('running_var', cranet.ones(self.num_features))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 cranet.tensor(0, dtype=cranet.long))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight = Parameter(np.ones(self.num_features))
            self.bias = Parameter(np.zeros(self.num_features))

    def _check_input_dim(self, inp: Tensor):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class _BatchNorm(_NormBase):
    def _check_input_dim(self, inp: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked += 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            x,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, inp: Tensor):
        if inp.dim() != 4:
            raise ValueError(f'expect 4-dim input, got {inp.dim()}-dim')

    def __repr__(self):
        return f"BatchNorm2d({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"
