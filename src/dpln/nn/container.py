from typing import List

from .module import Module
from .optimizer import Optimizer


class Sequential(Module):
    def __init__(self, *args: Module):
        super().__init__()
        self.layers: List[Module] = [i for i in args]

    def add(self, layer: Module):
        self.layers.append(layer)

    def forward(self, x):
        z = x
        for i in self.layers:
            z = i.forward(z)
        return z

    def backward(self, delta):
        for i in self.layers[::-1]:
            delta = i.backward(delta)
        return delta

    def update(self, optimizer: Optimizer):
        for i in self.layers:
            i.update(optimizer)
