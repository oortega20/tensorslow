from abc import ABC
from typing import Tuple

from tensorslow.linalg import Tensor


loss = float


class Model(ABC):
    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        for layer in self.layers[:-1]:
            x = layer(x)
        batch_loss, grad = self.layers[-1](x, y)
        return batch_loss, grad

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        return self.forward(x, y)

    def backward(self, grad: Tensor) -> Tensor:
        for layer in self.layers[:-1][::-1]:
            grad = layer.backward(grad)
        return grad
