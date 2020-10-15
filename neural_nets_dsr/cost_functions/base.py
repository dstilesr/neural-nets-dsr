import numpy as np
from typing import Callable
F = Callable[[np.ndarray, np.ndarray], float]


class CostFunction(F):
    """
    Base class to represent cost functions.
    """

    def __init__(
            self,
            function: F,
            gradient: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """

        :param function: Cost function.
        :param gradient: Function to compute derivatives of the cost function.
        """
        self.__function = function
        self.__gradient = gradient

    @property
    def gradient(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Gradient (Read only).
        :return:
        """
        return self.__gradient

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the cost on the given set of predictions.
        :param y_true:
        :param y_pred:
        :return:
        """
        return self.__function(y_true, y_pred)
