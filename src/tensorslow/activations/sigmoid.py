import math

from tensorslow.linalg import Tensor 
from tensorslow.activations import Activation

f_x = lambda x: 1 / (1 + (math.e ** (-1 * x)))
df_x = lambda x: f_x(x) * (1 - f_x(x))

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(f_x, df_x)

