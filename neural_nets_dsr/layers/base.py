import numpy as np
from abc import ABC, abstractmethod


class BaseLayer(ABC):
    """
    Base class for layers of a network.
    """

    @abstractmethod
    def forward_prop(
            self,
            x: np.ndarray,
            keep_cache: bool = False) -> np.ndarray:
        """
        Forward Propagation.
        :param x:
        :param keep_cache:
        :return:
        """
        pass

    @abstractmethod
    def back_prop(self, da: np.ndarray):
        """
        Compute derivatives by back propagation.
        :param da:
        :return:
        """
        pass
