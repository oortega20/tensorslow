from typing import Tuple
from math import log, log2

from tensorslow.losses import Loss
from tensorslow.linalg import Tensor

loss = float


class CrossEntropyLoss(Loss):
    """
    Compute Cross-Entropy Loss and Gradient:
    for a dataset with c classes
    the cross entropy loss for a probability
    distribution is as follows:

    CrossEntropy(y_hat, y) = -sum(i=1...k (y_i log(y_hat_i)))
    where y_i is the indicator function for whether sample_i is
    a member of class i
    and y_hat_i is the probability that sample i is a member of
    class i.
    """
    def __init__(self, units='nats'):
        self.loss = 0
        self.grad = Tensor([], (1,))
        self.log = log if units == 'nats' else log2

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        """
        Compute loss with respect to y_hat and y
        :param y_hat: model predictions
        :param y: true labels
        :return: loss and gradient of loss
        """
        if y_hat.order != 2 or y.order != 1 or y_hat.shape[0] != y.shape[0]:
            raise ValueError(f'invalid shapes for y_hat and y: {y_hat.shape} {y.shape}')
        num_samples, _ = y_hat.shape
        loss, arg_max = 0.0, Tensor([], y_hat.shape, init='zeros')
        for n in range(num_samples):
            y_class = y.tensor[n]
            loss -= self.log(y_hat.tensor[n][y_class])
            arg_max.tensor[n][y_class] = 1

        self.loss = loss
        self.grad = (y_hat - arg_max)
        return self.loss, self.grad
