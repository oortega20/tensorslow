# Tensorslow

![Tensorslow Logo](imgs/tensorslow.jpg)


Have you ever wanted to to use the power of neural network modeling with absolutely **NONE** of the rush given by competing neural network frameworks? Well if you answered yes to the previous question then Tensorslow is the machine learning framework to use. Using state of the art [**Python3**](https://www.python.org/doc/humor/#the-zen-of-python) lists rather than any complicated and vectorized backends from languages of yester-year like  [C++](https://en.wikipedia.org/wiki/C%2B%2B), we seek to use the Zen of Python to make your machine learning models learn at a pace more compatible with that of your three remaining brain-cells (or at least mine). So, get ready to sit back, take a sip of that green-tea, and enjoy the magic of machine learning with absolutely **NONE** of the stress provided by a machine that works way too quickly for its own good.

# Install

To install the current release of Tensorslow
```shell
python -m pip install -e tensorslow
```

# Try your first Tensorslow Program

Forward pass of simple dense layer with `input=3`, `weight=3`, and `bias=1` 
```python
>>> from tensorslow.layers import Dense

>>> x = Dense([3], [3], [1])
>>> x.forward()
10
```
