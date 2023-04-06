from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class SGDMomentum(Optimizer):
    """
    Tensorslow Stochastic Gradient Descent Optimizer with Momentum
    Given a loss function with learning-rate lambda and momentum parameter gamma
    we can describe the recurrence relation for SGD with momentum as follows:
    w_n+1 = w_t - v_n+1
    v_n+1 = gamma * v_n + lambda * grad_w(L(w))
    ...
    v_0 = 0
    """
    def __init__(self, model, learning_rate=1e-3, momentum=0.99):
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
        """
        Perform SGD with Momentum update rule
        :param layer: layer in neural network model
        :return: None
        """
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            weight = weights[grad_name]
            v = self.velocities[name][grad_name]
            v_new = self.momentum * v + (1 - self.momentum) * grad
            self.velocities[name][grad_name] = v_new
            weights[grad_name] = weight - self.learning_rate * v_new
