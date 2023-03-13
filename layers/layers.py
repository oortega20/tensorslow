from abc import ABC
from abc import abstractmethod

from tensorslow.linalg import Tensor

class Layer(ABC):
    def __init__(self): 
        self.grads = dict()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, dout: Tensor) -> Tensor:
        pass
