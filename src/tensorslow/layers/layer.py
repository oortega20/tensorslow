from abc import ABC
from abc import abstractmethod

from tensorslow.linalg import Tensor


class Layer(ABC):
    """Abstract class for dataset objects"""
    def __init__(self, name: str):
        self.name = name
        self.weights = dict()
        self.grads = dict()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward propagation on a tensor.
        :param x: Input Tensor
        :return: Output Tensor
        """
        pass

    @abstractmethod
    def backward(self, dout: Tensor) -> Tensor:
        """
        Perform back-propagation on a tensor dout.
        :param dout: the incoming gradient needed to perform back-propagation.
        :return: gradient of loss with respect to the current layer.
        """
        pass
