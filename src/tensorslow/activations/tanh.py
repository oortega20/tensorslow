import math

from tensorslow.activations import Activation

f = lambda x: (math.e ** x - math.e ** (-1 * x)) / (math.e ** x + math.e ** (-1 * x))
df = lambda x: 1 - f(x) ** 2


class Tanh(Activation):
    """
    Activation function softmax:
    For an input_vector x
    tanh(x) = (e^2x - 1) / (e^2x + 1)
    """
    def __init__(self):
        super().__init__(f, df)
