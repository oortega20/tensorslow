from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class RMSProp(Optimizer):
    def __init__(self, model, learning_rate=1e-3, momentum=0.99, epsilon=1e-7):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        self.velocities = dict()
        for layer in model.layers:
            if isinstance(layer, Layer):
                self.velocities[layer.name] = dict()
                for w_name, w in layer.weights.items():
                    self.velocities[layer.name][w_name] = Tensor([], w.shape, init='zeros')

    def update_rule(self, layer):
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            v = self.velocities[name][grad_name]
            v_new = self.momentum * v + ((1 - self.momentum) * (grad ** 2))
            delta_w = (self.learning_rate / (v_new.sqrt() + self.epsilon) * grad)
            weights[grad_name] = weights[grad_name] - delta_w
            self.velocities[name][grad_name] = v_new


