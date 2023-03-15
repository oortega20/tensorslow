import math

from tensorslow.linalg import Tensor
from tensorslow.activations import Activation


def f(x: Tensor) -> Tensor:
    if not x.order == 2:
        raise ValueError('Softmax activation only for tensors of order 2')
    s = x.max(axis=1)
    print(s, s.shape)
    s = s.expand_dims(axis=0)
    print(s, s.shape)
    print(x - s, (x - s).shape)
    e_x = math.e ** (x - s)
    div = e_x.sum(axis=1)
    div = div.expand_dims(axis=1)
    return e_x / div



class Softmax(Activation):
    def __init__(self):
        super().__init__(f, f)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return self.function(x)

    def backward(self, dout: Tensor) -> Tensor:
        grad = self.derivative(self.x)
        return grad * dout
