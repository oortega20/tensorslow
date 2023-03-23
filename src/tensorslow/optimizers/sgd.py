from tensorslow.optimizers import Optimizer
from tensorslow.models import Model


class SGD(Optimizer):
    """
    Tensorslow Stochastic Gradient Descent Optimizer
    Given a loss function with learning-rate lambda
    we can describe the recurrence relation for SGD as follows:
    w_n+1 = w_n - lambda * grad_w(L(w))
    """
    def __init__(self, model: Model, learning_rate=1e-4):
        super().__init__(model)
        self.learning_rate = learning_rate

    def update_rule(self, layer):
        """
        Perform SGD update rule
        :param layer: layer in neural network model
        :return: None
        """
        weights, grads = layer.weights, layer.grads
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            new_weight = weight - (self.learning_rate * grad)
            weights[grad_name] = new_weight

