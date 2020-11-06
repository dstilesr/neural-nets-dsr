import numpy as np
from .base import UpdateStrategy


class GradientDescentStrategy(UpdateStrategy):
    """
    Compute updates by gradient descent.
    """
    NAME: str = "gradient_descent"

    def __init__(self, learning_rate: float):
        assert learning_rate > 0.0
        self.__learning_rate = learning_rate

    @property
    def lr(self) -> float:
        return self.__learning_rate

    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Updates parameters by gradient descent.
        :param vals:
        :param grad:
        :return:
        """
        assert vals.shape == grad.shape
        return vals - self.lr * grad
