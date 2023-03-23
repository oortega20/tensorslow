from typing import Tuple

from tensorslow.losses import Loss
from tensorslow.linalg import Tensor

loss = float


class L1Loss(Loss):
    """
    Compute L1 Loss and Gradient:
    l1(y_hat, y) = | y_hat - y |
    """
    def __init__(self):
        self.loss = 0
        self.grad = Tensor([], (1,))

    def compute_loss(self, y_hat: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        """
        Compute loss with respect to y_hat and y
        :param y_hat: model predictions
        :param y: true labels
        :return: loss and gradient of loss
        """
        if y_hat.shape != y.shape:
            raise ValueError(f'x shape not compatible with y shape: {y_hat.shape} {y.shape}')

        residuals = (y_hat - y).abs()
        self.loss = residuals.sum()

        self.grad = residuals.unary_op(lambda x: 1 if x > 0 else -1)
        return self.loss, self.grad
