from .module import Module
from src.dpln.autograd.tensor import Tensor

from typing import (
    List,
    Union,
)


class Sequential(Module):

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.layers: List[Module] = [i for i in modules]

    def add_module(self, layer: Module) -> None:
        self.layers.append(layer)

    def forward(self, input: Tensor) -> Tensor:
        for module in self.layers:
            input_ = module(input_)
        return input_

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, List[Module]]:
        cls_ = type(self)
        if isinstance(idx, slice):
            return cls_(self.layers[idx])
        elif isinstance(idx, int):
            return self.layers[idx]
        else:
            msg = '{cls.__name__} indices must be integer or slice object'
            raise TypeError(msg.format(cls=cls_))

    def __repr__(self) -> str:
        ...
