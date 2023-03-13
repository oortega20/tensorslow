from abc import ABC
from abc import abstractmethod

from tensorslow.linalg import Tensor

class Activation(ABC):
    def __init__(self, fn):
        self.function = fn 
        self.x = None

    def forward(self, x: Tensor) -> Tensor: 
        out = x.unary_op(self.function)
        self.grad = Tensor([], x.shape, init='ones')
        return out

    def __call__(self, x: Tensor) -> Tensor:
        self.x = x
        return self.forward(x)

    @abstractmethod
    def backward(self, dout:Tensor) -> Tensor:
        pass
