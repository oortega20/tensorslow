import numpy as np

from tensorslow.linalg import Tensor
from tensorslow.activations import Softmax

s = Softmax()
x = Tensor(list(range(12)), (3,4))
dout = Tensor([], (3,4), init='ones')
def gradient_check(x, epsilon=1e-7):
    x_plus = x + epsilon
    x_minus = x - epsilon

    J_plus = s(x_plus)
    J_minus = s(x_minus)

    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    # Check if gradapprox is close enough to backward propagation
    grad = s.backward(dout)

    numerator = np.linalg.norm((grad - gradapprox))
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print('The gradient is correct')
    else:
        print('The gradient is wrong')

    return difference