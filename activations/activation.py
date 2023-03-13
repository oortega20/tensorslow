from abc import ABC
from abc import abstractmethod

from tensorslow.linalg import Tensor

class Activation(ABC):
    def __init__(fn):
        self.function = fn 
        self.grad = None

    def forward(self, x: Tensor) -> Tensor: 
        out = x.apply_unary(self.function)
        self.grad = Tensor([], x.shape, init='ones')
        return out

    @abstractmethod
    def backward(self, dout:Tensor) -> Tensor:
        pass
