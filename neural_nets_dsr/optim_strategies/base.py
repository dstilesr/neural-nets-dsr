import numpy as np
from abc import ABC, abstractmethod


class UpdateStrategy(ABC):
    """
    Base class for parameter update strategies.
    """

    NAME: str

    @abstractmethod
    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Compute updated parameter values.
        :param vals: Initial values.
        :param grad: Gradient.
        :return:
        """
        pass
