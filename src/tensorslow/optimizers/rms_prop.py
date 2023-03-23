from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class RMSProp(Optimizer):
    """
    Tensorslow Root-Mean-Squared Propagation Optimizer
    Given a loss function with learning-rate lambda and momentum parameter gamma
    we can describe the recurrence relation for SGD with nesterov-momentum as follows:
          w_n+1 = w_n - delta_w_n+1
    delta_w_n+1 = lambda / (sqrt(v_n+1)) * grad_w(L(w))
          v_n+1 = gamma * v_n + (1 - gamma)* grad_w(L(w))^2
    ...
          v_0 = 0
    """
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
        """
        Perform RMSProp update rule
        :param layer: layer in neural network model
        :return: None
        """
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            v = self.velocities[name][grad_name]
            v_new = self.momentum * v + ((1 - self.momentum) * (grad ** 2))
            delta_w = (self.learning_rate / (v_new.sqrt() + self.epsilon) * grad)
            weights[grad_name] = weights[grad_name] - delta_w
            self.velocities[name][grad_name] = v_new


