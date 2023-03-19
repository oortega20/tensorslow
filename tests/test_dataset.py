import os
os.system('py -m pip install -e ../../tensorslow')

from tensorslow.datasets import MNIST
x = MNIST()
x_train, y_train = x.get_train_data()
x_test, y_test = x.get_test_data()

for x, y in list(zip(x_train, y_train)):
    print(x, y)


