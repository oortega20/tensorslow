from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class SGDNesterov(Optimizer):
    def __init__(self, model, learning_rate=1e-3, momentum=0.9):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = dict()
        for layer in model.layers:
            if isinstance(layer, Layer):
                self.velocities[layer.name] = dict()
                for w_name, w in layer.weights.items():
                    self.velocities[layer.name][w_name] = Tensor([], w.shape, init='zeros')

    def update_rule(self, layer):
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            v = self.velocities[name][grad_name]
            v_new = weight - (self.learning_rate * grad)
            weights[grad_name] = v_new + (self.momentum * (v_new - v))
            self.velocities[name][grad_name] = v_new
