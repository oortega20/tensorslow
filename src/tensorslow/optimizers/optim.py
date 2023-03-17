from abc import ABC
from abc import abstractmethod

from tensorslow.models import Model
from tensorslow.layers import Layer


loss = float


class Optimizer(ABC):
    def __init__(self, model):
        self.model = model

    def update(self):
        for layer in self.model.layers:
            if isinstance(layer, Layer):
                self.update_rule(layer.weights, layer.grads)

    @abstractmethod
    def update_rule(self, weights, grads):
        pass

