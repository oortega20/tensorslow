import math
from iteration_utilities import deepflatten

from tensorslow.linalg import Tensor
from tensorslow.activations import Activation


class Softmax(Activation):
    def __init__(self):
        self.x = None

    def function(self, x: Tensor) -> Tensor:
        if not x.order == 2:
            raise ValueError('Softmax activation only for tensors of order 2')
        s = x.max(axis=1)
        s = s.expand_dims(axis=1)
        e_x = math.e ** (x - s)
        div = e_x.sum(axis=1)
        div = div.expand_dims(axis=1)
        return e_x / div

    def derivative(self, dout: Tensor) -> Tensor:
        if not dout.order == 2:
            raise ValueError('Softmax derivative only for tensors of order 2')

        sm = Tensor.diagflat(list(deepflatten(self.x.tensor)))
        return (sm - self.x @ self.x.T) @ dout

    def forward(self, x: Tensor) -> Tensor:
        self.x = self.function(x)
        return self.x

    def backward(self, dout: Tensor) -> Tensor:
        return self.derivative(dout)
