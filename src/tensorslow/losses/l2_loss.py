from typing import Tuple

from tensorslow.losses import Loss
from tensorslow.linalg import Tensor

loss = float


class L2Loss(Loss):
    def __init__(self):
        self.loss = 0
        self.grad = Tensor([], (1,))

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        if y_hat.shape != y.shape:
            raise ValueError(f'x shape not compatible with y shape: {y_hat.shape} {y.shape}')

        residuals = ((y_hat - y) / 2) ** 2
        self.loss = residuals.sum()

        self.grad = (y_hat - y)
        return self.loss, self.grad
