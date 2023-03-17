from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor


class ADAM(Optimizer):
    def __init__(self, model, learning_rate=1e-3, beta_1=0.99, beta_2=0.9):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.velocities = dict()

    def update_rule(self, layer):
        pass
