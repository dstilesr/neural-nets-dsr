import numpy as np
from typing import Union, Tuple
from .gradient_descent import GradientDescent
from ..cost_functions.base import CostFunction


class GradientDescentL2(GradientDescent):
    """
    Gradient descent with L2 regularization.
    """

    def __init__(
            self,
            cost_func: Union[str, CostFunction],
            epochs: int = 700,
            batch_size: int = -1,
            axis: int = 1,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            verbose: bool = False):
        """

        :param cost_func:
        :param epochs:
        :param batch_size:
        :param axis:
        :param learning_rate:
        :param l2_param:
        :param verbose:
        """
        super().__init__(
            cost_func=cost_func,
            epochs=epochs,
            batch_size=batch_size,
            axis=axis,
            learning_rate=learning_rate,
            verbose=verbose
        )
        assert l2_param >= 0.0
        self.__l2_param = l2_param

    @property
    def l2_param(self) -> float:
        """
        L2 regularization parameter.
        :return:
        """
        return self.__l2_param

    def get_updates(
            self,
            w: np.ndarray,
            b: np.ndarray,
            dw: np.ndarray,
            db: np.ndarray,
            lyr_index: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute parameter updates for GD with L2 regularization.
        :param w:
        :param b:
        :param dw:
        :param db:
        :param lyr_index:
        :return:
        """
        wgrad = self.learning_rate * (dw + self.l2_param * w)
        bgrad = self.learning_rate * db
        return w - wgrad, b - bgrad
