import os
import numpy as np

os.system('py -m pip install -e ../../tensorslow')
from tensorslow.activations import Softmax
from tensorslow.linalg import Tensor
from tensorslow.losses import CrossEntropyLoss

s = Softmax()
c = CrossEntropyLoss()
N, C = (6, 2)
x = Tensor(list(range(12)), (N, C))
y = Tensor([0, 1, 0, 0, 1, 1], (N,))
print(x, 'x')
y_hat = s(x)
print(y_hat, 'softmax(x)')
loss, grad = c(y_hat, y)

print(grad, 'dloss / softmax(x)')
dloss_ds = s.backward(grad)
print(dloss_ds, 'dloss / dx = dloss / softmax(x) * d softmax(x) / dx')

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


print(softmax(np.array(x.tensor)), 'numpy softmax')

