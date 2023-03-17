from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor


class RMSProp(Optimizer):
    def __init__(self, model, learning_rate=1e-3, momentum=0.99):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = dict()

    def update_rule(self, layer):
        pass
