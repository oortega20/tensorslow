from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class Dense(Layer):
    def __init__(self, name: str, in_dim=1, out_dim=1,  w_init='xavier', b_init='ones'):
        super().__init__(name)
        w = Tensor([], init=w_init, shape=(in_dim, out_dim))
        b = Tensor([], init=b_init, shape=(out_dim,))
        w = w / w.num_entries
        b = b / b.num_entries
        self.x = None
        self.dout = None
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
        self.dout = dout
        self.grads['w'] = self.x.T @ dout
        self.grads['b'] = Tensor([], self.weights['b'].shape, init='ones') * dout.sum(axis=0)
        dx = dout @ self.weights['w'].T
        return dx


