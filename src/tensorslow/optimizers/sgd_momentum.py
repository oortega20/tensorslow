from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor


class SGDMomentum(Optimizer):

    def __init__(self, model, learning_rate=1e-3, momentum=0.99):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = dict()

    def update_rule(self, layer):
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            if not self.velocities.get(name):
                self.velocities[name] = dict()
            if not self.velocities.get(name).get(grad_name):
                self.velocities[name][grad_name] = Tensor([], weight.shape, init='zeros')
            v = self.velocities[name][grad_name]
            v_new = self.momentum * v + self.learning_rate * grad
            self.velocities[name][grad_name] = v_new
            weights[grad_name] = weight - v_new
