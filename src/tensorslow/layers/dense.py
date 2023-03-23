from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class Dense(Layer):
    """Dense (or Affine Layer) will perform the computation
    Dense(X) = XW + b
    Where X is of dimension : (num_samples, in_dim)
          W is of dimension : (in_dim, out_dim)
          b is of dimension : (out_dim,)
    """
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
        """
        Perform forward propagation on a tensor for an affine layer.
        :param x: Input Tensor
        :return: Output Tensor
        """
        self.x = x
        w = self.weights['w']
        b = self.weights['b']
        return (x @ w) + b

    def backward(self, dout: Tensor) -> Tensor:
        """
        Perform back-propagation on a tensor dout.
        :param dout: the incoming gradient needed to perform back-propagation.
        :return: gradient of loss with respect to the current layer.
        """
        self.dout = dout
        self.grads['w'] = self.x.T @ dout
        self.grads['b'] = Tensor([], self.weights['b'].shape, init='ones') * dout.sum(axis=0)
        dx = dout @ self.weights['w'].T
        return dx


