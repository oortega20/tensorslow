import dill
import pickle
from abc import ABC
from typing import Tuple, Optional

from tensorslow.linalg import Tensor


loss = float


class Model(ABC):
    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x: Tensor, y: Optional[Tensor]=None):
        for layer in self.layers[:-1]:
            x = layer(x)

        if y:
            batch_loss, grad = self.layers[-1](x, y)
        else:
            return x
        return batch_loss, grad

    def __call__(self, x: Tensor, y: Optional[Tensor]) -> Tuple[loss, Tensor]:
        return self.forward(x, y=y)

    def backward(self, grad: Tensor) -> Tensor:
        for layer in self.layers[:-1][::-1]:
            grad = layer.backward(grad)
        return grad

    def save(self, path: str='model.pkl'):
        with open(path, 'wb') as f:
            dill.dump(self, f)


