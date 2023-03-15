from tensorslow.linalg import Tensor
from tensorslow.layers import Layer

class Dense(Layer):
    def __init__(self, w_init='ones', b_init='ones', in_shape=(10,10), out_dims=5):
        in_r, in_c = in_shape
        self.w = Tensor([], init=w_init, shape=(in_c, out_dims))
        self.b = Tensor([], init=b_init, shape=(in_r, out_dims))
        self.grad = dict()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return (self.x @ self.w) + self.b

    def backward(self, dout: Tensor) -> Tensor:
        self.grad['w'] = self.x.T @ dout 
        self.grad['b'] = Tensor([], init='ones', shape=dout.shape) * dout 
        self.grad['x'] = dout @ self.w
        return self.grad['x']


