from abc import ABC
from abc import abstractmethod

from tensorslow.linalg import Tensor

class Activation(ABC):
    def __init__(data: Tensor):
        self.data = data
        self.grad = Tensor(shape=data.shape, init='zeros')

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, x:) -> Tensor:
        pass
