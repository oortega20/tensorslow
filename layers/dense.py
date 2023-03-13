from tensorslow.linalg import Tensor
from tensorslow.layers import Layer

class Dense(Layer):
    def __init__(self, w_init, b_init, w_shape):
        self.w = Tensor([], init=w_init, shape=w_shape)
        self.b = Tensor([], init=b_init, shape=w_shape[1:])

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return self.x @ self.w + self.b

    def backward(self, dout: Tensor) -> Tensor:
        return dout #<TODO: need to work out the math for this

