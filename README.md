# Tensorslow

![Tensorslow Logo](imgs/tensorslow.jpg)


Have you ever wanted to to use the power of neural network modeling with absolutely **NONE** of the rush given by competing neural network frameworks? Well if you answered yes to the previous question then Tensorslow is the machine learning framework to use. Using state of the art [**Python3**](https://www.python.org/doc/humor/#the-zen-of-python) lists rather than any complicated and vectorized backends from languages of yester-year like  [C++](https://en.wikipedia.org/wiki/C%2B%2B), we seek to use the Zen of Python to make your machine learning models learn at a pace more compatible with that of your three remaining brain-cells (or at least mine). So, get ready to sit back, take a sip of that green-tea, and enjoy the magic of machine learning with absolutely **NONE** of the stress provided by a machine that works way too quickly for its own good.

# Install
First download the code from the repository
```shell
git clone git@github.com:oortega20/tensorslow.git
```

To install the current release of Tensorslow
```shell
python -m pip install -e tensorslow
```

# Try some simple Tensorslow Programs
Some fun matrix manipulations with tensorslow's linear-algebra package 
```python
>>> from tensorslow.linalg import Tensor
>>> x = Tensor(list(range(6)), (2,3))
>>> x
Tensor([[0.000 1.000 2.000]
        [3.000 4.000 5.000]])
>>> x @ x.T
Tensor([[5.000 14.000]
        [14.000 50.000]])
>>> x.T @ x
Tensor([[9.000 12.000 15.000]
        [12.000 17.000 22.000]
        [15.000 22.000 29.000]])
>>> x - 3
Tensor([[-3.000 -2.000 -1.000]
        [0.000 1.000 2.000]])
>>> 0 * x
Tensor([[0.000 0.000 0.000]
        [0.000 0.000 0.000]])
```
A simple demonstration of forward propagation with Tensorslow's available layers and activations

```python
>>> from tensorslow.linalg import Tensor
>>> from tensorslow.activations import Relu
>>> from tensorslow.layers import Dense
>>>
>>> x = Tensor(list(range(6)), (2,3))
>>> x
Tensor([[0.000 1.000 2.000]
        [3.000 4.000 5.000]])
>>> act = Sigmoid()
>>> f = Dense('f', in_dim=3, out_dim=3)
>>> f.weights['w']
Tensor([[0.057 0.051 0.021]
        [0.047 -0.031 0.003]
        [0.015 -0.052 0.058]])
>>> f.weights['b']
Tensor([0.333 0.333 0.333])
>>> out = act(f(x))
>>> out
Tensor([[0.601 0.549 0.611]
        [0.682 0.525 0.667]])
>>>
```
Inference using Tensorslow MNIST Classifier
```python
from tensorslow.datasets import MNIST
from tensorslow.models import ts_mnist_classifier

model = ts_mnist_classifier(from_ts=True)
data = MNIST(load_train=False, load_test=True, batch_size=128)
x_test, y_test = data.get_test_data()
for x, y in zip(x_test, y_test):
    probs = model.forward(x) # if we only want the class probabilities
    loss, grad = model.forward(x, y) # if we want to compute losses and gradients
```

Example of simple model training using tensorslow

```python
from tensorslow.datasets import MNIST
from tensorslow.models import ts_mnist_classifier
from tensorslow.optimizers import SGD


data = MNIST(batch_size=128)
x_train, y_train = data.get_train_data()
epochs = 10
display_batch = 10
save_model = 100
model = ts_mnist_classifier(from_ts=False)


opt = SGD(model, learning_rate=5e-4)
train_loss, test_loss = [], []
for epoch in range(epochs):
    for data in zip(x_train, y_train):
        x, y = data
        batch_loss, grad = model(x, y)
        model.backward(grad)
        opt.update()
```
Details of the Tensorslow MNIST Classifier can be found in a link to our [tensorslow-experimentation repo](https://github.com/oortega20/tensorslow-experimentation). Improving the accuracy of the current mnist classifier is something being researched! 
       

