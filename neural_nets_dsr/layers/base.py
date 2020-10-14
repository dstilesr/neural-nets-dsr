import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from ..activations import ActivationFunc, ACTIVATIONS_NAMES


class BaseLayer(ABC):
    """
    Base class for layers of a network.
    """

    @staticmethod
    def get_activation(
            activation: Union[ActivationFunc, str]) -> ActivationFunc:
        """
        Get an activation function instance from its name.
        :param activation: Name of an activation function (or the function
            itself).
        :return:
        """
        if isinstance(activation, ActivationFunc):
            act = activation
        elif activation in ACTIVATIONS_NAMES.keys():
            act = ACTIVATIONS_NAMES[activation]
        else:
            raise ValueError("Unknown activation function!")
        return act

    @abstractmethod
    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward Propagation.
        :param x:
        :param train_mode:
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

    @abstractmethod
    def set_weights(self, *args, **kwargs):
        """
        Set new values to the layer's weights.
        :param args:
        :param kwargs:
        :return:
        """
        pass
