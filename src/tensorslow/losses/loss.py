from abc import ABC
from abc import abstractmethod
from typing import Tuple

from tensorslow.linalg import Tensor


loss = float


class Loss(ABC):
    def __init__(self):
        self.loss = 0
        self.grad = Tensor([], (1,))

    @abstractmethod
    def compute_loss(self, x: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        pass

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[loss, Tensor]:
        return self.compute_loss(x, y)
