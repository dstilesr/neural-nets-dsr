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
            gradient: Callable[[np.ndarray, np.ndarray], np.ndarray],
            name: str):
        """

        :param function: Cost function.
        :param gradient: Function to compute derivatives of the cost function.
        :param name: Name of the cost function
        """
        if len(name) == 0:
            raise ValueError("Name must be non-empty!")
        self.__function = function
        self.__gradient = gradient
        self.__name = name

    @property
    def gradient(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Gradient (Read only).
        :return:
        """
        return self.__gradient

    @property
    def name(self) -> str:
        """
        Name of the cost function.
        :return:
        """
        return self.__name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the cost on the given set of predictions.
        :param y_true:
        :param y_pred:
        :return:
        """
        return self.__function(y_true, y_pred)
