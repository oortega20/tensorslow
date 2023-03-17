import os
os.system('py -m pip install -e ../../tensorslow')

from tensorslow.models import ts_mnist_classifier
from tensorslow.models import ts_l1_regressor
from tensorslow.models import ts_l2_regressor
from tensorslow.linalg import Tensor
from tensorslow.optimizers import SGD

x_regress = Tensor(list(range(10)), (10, 1))
x_mnist = Tensor(list(range(28 * 28)), (1, 28 * 28))
y_regress = Tensor(list(range(10)), (10, 1))
y_mnist = Tensor([1], (1,))

mnist = ts_mnist_classifier()
l1 = ts_l1_regressor()
l2 = ts_l2_regressor()
opt_m = SGD(mnist)
opt_l1 = SGD(l1)
opt_l2 = SGD(l2)
num_iters = 100
print('currently training an l1regressor, l2regressor, and an mnist model with a dummy sample data')
for _ in range(num_iters):
    print(_, 'current iter')
    l1loss, l1grad = l1(x_regress, y_regress)
    l2loss, l2grad = l2(x_regress, y_regress)
    celoss, cegrad = mnist(x_mnist, y_mnist)
    l1.backward(l1grad)
    l2.backward(l2grad)
    mnist.backward(cegrad)
    opt_m.update()
    opt_l2.update()
    opt_l1.update()
    print(l1loss, l2loss, celoss, 'if the losses are going down, then we did bois, tensorslow trainss')