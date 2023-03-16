from abc import ABC

from tensorslow.linalg import Tensor


class Activation(ABC):
    def __init__(self, fn, d_fn):
        self.function = fn 
        self.derivative = d_fn
        self.x = None

    def forward(self, x: Tensor) -> Tensor: 
        out = x.unary_op(self.function)
        self.x = x
        return out

    def __call__(self, x: Tensor) -> Tensor:
        self.x = x
        return self.forward(x)

    def backward(self, dout: Tensor) -> Tensor:
        grad = self.x.unary_op(self.derivative)
        return grad * dout 
