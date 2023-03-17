from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class Dense(Layer):
    def __init__(self, in_dim=1, out_dim=1,  w_init='ones', b_init='ones'):
        w = Tensor([], init=w_init, shape=(in_dim, out_dim))
        b = Tensor([], init=b_init, shape=(out_dim,))
        self.x = None
        self.grads = dict(w=None, b=None)
        self.weights = dict(w=w, b=b)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        w = self.weights['w']
        b = self.weights['b']
        return (x @ w) + b

    def backward(self, dout: Tensor) -> Tensor:
        self.grads['w'] = self.x.T @ dout
        self.grads['b'] = Tensor([], self.weights['b'].shape, init='ones') * dout.sum(axis=0)
        dx = dout @ self.weights['w'].T
        return dx


