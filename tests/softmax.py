import os
import numpy as np

os.system('py -m pip install -e https://github.com/oortega20/tensorslow.git')
from tensorslow.activations import Softmax
from tensorslow.linalg import Tensor
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = np.expand_dims(s, axis=1)
    print(z - s, (z - s).shape)
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = np.expand_dims(div, axis=1)
    return e_x / div

x1 = np.array(list(range(12))).reshape((3,4))

r = softmax(x1)

s = Softmax()

x = Tensor(list(range(12)), (3, 4))
s(x)
