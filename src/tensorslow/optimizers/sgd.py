from tensorslow.optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__(model)
        self.learning_rate = learning_rate

    def update_rule(self, weights, grads):
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            new_weight = weight - self.learning_rate * grad
            weights[grad_name] = new_weight

