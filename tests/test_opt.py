import os
os.system('py -m pip install -e ../../tensorslow')

from tensorslow.models import ts_l2_regressor
from tensorslow.linalg import Tensor
from tensorslow.optimizers import SGD
from tensorslow.optimizers import SGDNesterov
from tensorslow.optimizers import SGDMomentum
from tensorslow.optimizers import RMSProp
from tensorslow.optimizers import ADAM


x_regress = Tensor(list(range(10)), (10, 1))
y_regress = Tensor(list(range(10,-1,-1)), (10, 1))


l2 = ts_l2_regressor()

sgd_l2 = SGDNesterov(l2, learning_rate=1e-3, momentum=.9)

num_iters = 10000

print('Testing all the optimizers')
for _ in range(num_iters):
    mloss, mgrad  = l2(x_regress, y_regress)
    print('model loss', mloss)
    l2.backward(mgrad)
    sgd_l2.update()

