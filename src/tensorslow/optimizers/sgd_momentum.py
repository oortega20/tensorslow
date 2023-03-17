from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class SGDMomentum(Optimizer):

    def __init__(self, model, learning_rate=1e-3, momentum=0.99):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = dict()
        for layer in model.layers():
            if isinstance(layer, Layer):
                self.velocities[layer.name] = dict()
                for grad_name, grad in layer.grads.items():
                    self.velocities[layer.name][grad_name] = Tensor([], grad.shape, init='zeros')

    def update_rule(self, layer):
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            v = self.velocities[name][grad_name]
            v_new = self.momentum * v + self.learning_rate * grad
            self.velocities[name][grad_name] = v_new
            weights[grad_name] = weight - v_new
