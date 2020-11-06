import numpy as np
from .base import UpdateStrategy


class GradientDescentL2Strategy(UpdateStrategy):
    """
    Compute updates by gradient descent with L2 regularization.
    """

    NAME: str = "gradient_descent_l2"

    def __init__(self, learning_rate: float, l2_param: float = 0.01):
        self.__learning_rate = learning_rate
        self.__l2_param = l2_param

    @property
    def lr(self) -> float:
        return self.__learning_rate

    @property
    def l2_param(self) -> float:
        return self.__l2_param

    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Update parameters by l2 gradient descent.
        :param vals:
        :param grad:
        :return:
        """
        assert vals.shape == grad.shape
        full_grad = grad + self.l2_param * vals
        return vals - self.lr * full_grad
