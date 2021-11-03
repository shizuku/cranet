from __future__ import annotations

from cranet import Tensor

from .module import Module

from collections import OrderedDict
from typing import (
    List,
    Dict,
    Union,
)


class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.register_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.register_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for (_, module) in self._modules.items():
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, Sequential]:
        if isinstance(idx, slice):
            return Sequential(*self.layers[idx])
        elif isinstance(idx, int):
            return self.layers[idx]
        else:
            msg = '{cls.__name__} indices must be integer or slice object'
            raise TypeError(msg.format(cls=Sequential))
