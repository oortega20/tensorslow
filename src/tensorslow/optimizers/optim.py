from abc import ABC
from abc import abstractmethod

from tensorslow.models import Model
from tensorslow.layers import Layer


class Optimizer(ABC):
    def __init__(self, model: Model):
        self.model = model

    def update(self):
        for layer in self.model.layers:
            if isinstance(layer, Layer):
                self.update_rule(layer)

    @abstractmethod
    def update_rule(self, layer):
        pass

