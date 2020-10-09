import numpy as np
from typing import Callable
from ..network import NeuralNet
from abc import ABC, abstractmethod


class Optimizer(Callable[[NeuralNet, np.ndarray, np.ndarray], NeuralNet], ABC):
    """
    Base class to model an optimizer for a neural network.
    """

    @abstractmethod
    def __call__(
            self,
            network: NeuralNet,
            x: np.ndarray,
            y: np.ndarray) -> NeuralNet:
        """
        Fits the given network.
        :param network:
        :return:
        """
        pass
