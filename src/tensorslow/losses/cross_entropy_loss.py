from typing import Tuple
from math import log, log2

from tensorslow.losses import Loss
from tensorslow.linalg import Tensor

loss = float


class CrossEntropyLoss(Loss):
    def __init__(self, units='nats'):
        self.loss = 0
        self.grad = Tensor([], (1,))
        self.log = log if units == 'nats' else log2

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        if y_hat.order != 2 or y.order != 1 or y_hat.shape[0] != y.shape[0]:
            raise ValueError(f'invalid shapes for y_hat and y: {y_hat.shape} {y.shape}')
        num_samples, _ = y_hat.shape
        loss, arg_max = 0.0, Tensor([], y_hat.shape, init='zeros')
        for n in range(num_samples):
            y_class = y.tensor[n]
            if y_hat.tensor[n][y_class] <= 0:
                raise ValueError(f'{y_hat.tensor[n][y_class]}: negative prob value')
            loss -= self.log(y_hat.tensor[n][y_class])
            arg_max.tensor[n][y_class] = 1

        self.loss = loss
        self.grad = (y_hat - arg_max)
        return self.loss, self.grad
