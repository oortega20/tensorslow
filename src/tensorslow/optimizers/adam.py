from tensorslow.optimizers import Optimizer
from tensorslow.linalg import Tensor
from tensorslow.layers import Layer


class ADAM(Optimizer):
    """
    Tensorslow Adaptive Moment Optimizer
    Given a loss function with learning-rate lambda
    beta_1, beta_2, and epsilon hyper-parameters,
    we can describe the recurrence relation for SGD as follows:
       w_n+1 = w_n - (lambda / sqrt(v_hat_n+1) + epsilon) * m_hat_n
    m_hat_n  = m_n / (1 - beta_1^n)
    v_hat_n =  v_n / (1 - beta_2^n)
        m_n = beta_1 * m_n-1 + (1 - beta_1) * grad_w(L(w))
        v_n = beta_2 * v_n-1 + (1 - beta_2) * grad_w(L(w)^2
    ...
        m_0 = 0
        v_0 = 0
    """
    def __init__(self, model, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.velocities = dict()
        self.momentums = dict()
        self.t = 1
        for layer in model.layers:
            if isinstance(layer, Layer):
                self.velocities[layer.name] = dict()
                self.momentums[layer.name] = dict()
                for w_name, w in layer.weights.items():
                    self.velocities[layer.name][w_name] = Tensor([], w.shape, init='zeros')
                    self.momentums[layer.name][w_name] = Tensor([], w.shape, init='zeros')

    def update_rule(self, layer):
        """
        Perform ADAM update rule
        :param layer: layer in neural network model
        :return: None
        """
        weights, grads, name = layer.weights, layer.grads, layer.name
        for grad_name, grad in grads.items():
            m = self.momentums[name][grad_name]
            v = self.velocities[name][grad_name]
            m_new = self.beta_1 * m + ((1 - self.beta_1) * grad)
            v_new = self.beta_2 * v + ((1 - self.beta_2) * (grad ** 2))
            m_hat = m_new / (1 - self.beta_1 ** self.t)
            v_hat = v_new / (1 - self.beta_2 ** self.t)
            delta_w = self.learning_rate * (m_hat / (v_hat.sqrt() + self.epsilon))
            weights[grad_name] = weights[grad_name] - delta_w
            self.velocities[name][grad_name] = v_new
            self.momentums[name][grad_name] = m_new
