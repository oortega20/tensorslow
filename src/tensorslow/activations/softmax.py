import math
from iteration_utilities import deepflatten

from tensorslow.linalg import Tensor
from tensorslow.activations import Activation


class Softmax(Activation):
    """
    Activation function softmax:
    For an input_vector x with c classes
    the probability given by the softmax function is as follows.
    softmax(x)_i= 1 / e^x_i / sum(j=1..k  e^x_j)
    Where x_i is the ith entry in x
    """
    def __init__(self):
        self.x = None

    def function(self, x: Tensor) -> Tensor:
        """
        Perform forward pass on a Tensor x
        :param x: input logits
        :return: return softmax class probabilities
        """
        if not x.order == 2:
            raise ValueError('Softmax activation only for tensors of order 2')
        s = x.max(axis=1)
        s = s.expand_dims(axis=1)
        e_x = math.e ** (x - s)
        div = e_x.sum(axis=1)
        div = div.expand_dims(axis=1)
        return e_x / div

    def derivative(self, dout: Tensor) -> Tensor:
        """
        Perform back-propagation on a tensor dout.
        :param dout: the incoming gradient needed to perform back-propagation.
        :return: gradient of activation function with respect to the activation.
        """
        if dout and not dout.order == 2:
            raise ValueError('Softmax derivative only for tensors of order 2')

        num_samples, num_classes = self.x.shape
        grad = Tensor([], (num_classes, num_classes), init='zeros')
        cache = None
        for n in range(num_samples):
            sm = Tensor.diagflat(self.x.tensor[n], cache=cache)
            cache = sm.tensor
            outer_prod = Tensor(self.x.tensor[n], (num_classes, 1))
            grad = grad + (sm - (outer_prod @ outer_prod.T))
        return grad if not dout else dout @ grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward propagation on a tensor.
        :param x: Input Tensor
        :return: Output Tensor
        """
        self.x = self.function(x)
        return self.x

    def backward(self, dout: Tensor=None) -> Tensor:
        """
        Perform back-propagation on a tensor dout.
        :param dout: the incoming gradient needed to perform back-propagation.
        :return: gradient of activation function with respect to the activation.
        """
        return self.derivative(dout)
