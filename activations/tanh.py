import math

from tensorslow.linalg import Tensor
from tensorslow.activations import Activation

f = lambda x: (math.e ** x - math.e ** (-1 * x)) / (math.e ** x + math.e ** (-1 * x))
df = lambda x: 1 - f(x) ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(f, df)
