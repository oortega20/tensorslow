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
>>> act = Relu()
>>> f = Dense(in_shape=(2,3), out_dims=3)
>>> f.w
Tensor([[1.000 1.000 1.000]
        [1.000 1.000 1.000]
        [1.000 1.000 1.000]])
>>> f.b
Tensor([[1.000 1.000 1.000]
        [1.000 1.000 1.000]])
>>> out = act(f(x))
>>> out
Tensor([[4.000 4.000 4.000]
        [13.000 13.000 13.000]])
>>>
```
