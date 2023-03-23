from abc import ABC
from abc import abstractmethod

from tensorslow.models import Model
from tensorslow.layers import Layer


class Optimizer(ABC):
    """Abstract Class for Tensorslow Optimizers"""
    def __init__(self, model: Model):
        self.model = model

    def update(self):
        """
        Update parameters in neural network model
        :return: None
        """
        for layer in self.model.layers:
            if isinstance(layer, Layer):
                self.update_rule(layer)

    @abstractmethod
    def update_rule(self, layer: Layer):
        """
        Perform updates according to optimizer's algorithm
        :param layer: layer in neural network model
        :return: None
        """
        pass

