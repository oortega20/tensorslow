from abc import ABC

from tensorslow.linalg import Tensor


class Activation(ABC):
    """Abstract class delineating a general format for activation functions"""
    def __init__(self, fn, d_fn):
        self.function = fn 
        self.derivative = d_fn
        self.x = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward propagation on a tensor.
        :param x: Input Tensor
        :return: Output Tensor
        """
        out = x.unary_op(self.function)
        self.x = x
        return out

    def __call__(self, x: Tensor) -> Tensor:
        self.x = x
        return self.forward(x)

    def backward(self, dout: Tensor) -> Tensor:
        """
        Perform back-propagation on a tensor dout.
        :param dout: the incoming gradient needed to perform back-propagation.
        :return: gradient of activation function with respect to the activation.
        """
        grad = self.x.unary_op(self.derivative)
        return grad * dout 
