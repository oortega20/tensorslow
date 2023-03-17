from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class ADAM(Optimizer):
    def __init__(self, model, learning_rate=1e-3, beta_1=0.99, beta_2=0.9, epsilon=1e-7):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.velocities = dict()
        self.momentums = dict()
        for layer in model.layers():
            if isinstance(layer, Layer):
                self.velocities[layer.name] = dict()
                self.momentums[layer.name] = dict()
                for grad_name, grad in layer.grads.items():
                    self.velocities[layer.name][grad_name] = Tensor([], grad.shape, init='zeros')

    def update_rule(self, layer):
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            v = self.velocities[name][grad_name]
            m = self.momentums[name][grad_name]
            v_new = self.beta_1 * v - (1 - self.beta_1) * grad
            m_new = self.beta_2 * m - (1 - self.beta_2) * grad ** 2
            delta_w = -1 * (self.learning_rate * (v_new / (m_new + self.epsilon).sqrt()) * grad)
            weights[grad_name] = weights[grad_name] + delta_w
            self.velocities[name][grad_name] = v_new
            self.momentums[name][grad_name] = m_new
