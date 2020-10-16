import numpy as np
from .. import cost_functions
from ..network import NeuralNet
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Union

U = Union[str, cost_functions.CostFunction]


class Optimizer(Callable[[NeuralNet, np.ndarray, np.ndarray], NeuralNet], ABC):
    """
    Base class to model an optimizer for a neural network.
    """

    @staticmethod
    def get_cost_func(cost_name: U) -> cost_functions.CostFunction:
        """
        Gets a cost function from its name.
        :param cost_name:
        :return:
        """
        if isinstance(cost_name, cost_functions.CostFunction):
            out = cost_name
        elif cost_name in cost_functions.AVAILABLE_COST_FUNCS:
            out = cost_functions.COST_NAMES[cost_name]
        else:
            raise ValueError("Unknown cost function!")
        return out

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

    @abstractmethod
    def get_updates(self, *args, **kwargs) -> Iterable[np.ndarray]:
        """
        Auxiliary to compute updated weights.
        :param args: Arguments TBD in subclasses.
        :param kwargs: Keyword arguments TBD in subclasses.
        :return:
        """
        pass
