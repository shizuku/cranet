from src.dpln.nn.parameter import Parameter
from ..nn.modules.module import Module
    
class SGD:
    def __init__(self, module: Module, lr: float = 0.01) -> None:
        if not issubclass(module, Module):
            raise ValueError("for a optimizer, the first parameter must be a `Module` object or it's subclass ")
        self.module = module
        self.lr = lr

    def step(self) -> None:
        for parameter in self.module.parameters():
            parameter -= parameter.grad * self.lr