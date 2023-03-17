from tensorslow.optimizers import Optimizer
from tensorslow.models import Model


class SGD(Optimizer):
    def __init__(self, model: Model, learning_rate=1e-3):
        super().__init__(model)
        self.learning_rate = learning_rate

    def update_rule(self, layer):
        weights, grads = layer.weights, layer.grads
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            new_weight = weight - self.learning_rate * grad
            weights[grad_name] = new_weight

