import numpy as np
from typing import Union
from ..cost_functions import CostFunction
from .gradient_descent import GradientDescent


class GradientDescentL2(GradientDescent):
    """
    Gradient descent with L2 regularization.
    """

    def __init__(
            self,
            cost_func: Union[str, CostFunction],
            max_iterations: int = 700,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            verbose: bool = False):
        """

        :param cost_func:
        :param max_iterations:
        :param learning_rate:
        :param l2_param:
        :param verbose:
        """
        super().__init__(
            cost_func=cost_func,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=verbose
        )
        assert l2_param >= 0.0
        self.__l2_param = l2_param

    @property
    def l2_param(self) -> float:
        return self.__l2_param

    def gradient_descent_iteration(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Does a single iteration of gradient descent
        :param x: Training set features.
        :param y: Training set labels.
        :return: Cost before the backprop.
        """
        if self._network is None:
            raise NotImplementedError("No network selected!")

        y_pred = self._network.compute_predictions(x, True)
        cost = self.cost_func(y, y_pred)
        da = self.cost_func.gradient(y, y_pred)

        for lyr in self._network.layers[::-1]:
            dw, db, da = lyr.back_prop(da)

            reg_w = self.l2_param * lyr.weights
            lyr.set_weights(
                w=lyr.weights - self.learning_rate * dw - reg_w,
                b=lyr.biases - self.learning_rate * db
            )
        return cost
