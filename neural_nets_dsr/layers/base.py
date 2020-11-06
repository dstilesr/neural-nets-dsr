import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from .. import optim_strategies as optim
from ..activations import ActivationFunc, ACTIVATIONS_NAMES


class BaseLayer(ABC):
    """
    Base class for layers of a network.
    """

    def __init__(self):
        self.__trainable = True

    @staticmethod
    def strategy_from_name(name: str):
        if name in optim.STRATEGY_NAMES.keys():
            return optim.STRATEGY_NAMES[name]
        else:
            raise KeyError(
                "Unknown optimization strategy! "
                + "Supported strategies are: "
                + ", ".join(optim.STRATEGY_NAMES.keys())
            )

    @abstractmethod
    def set_update_strategy(self, strategy_name: str, **kwargs):
        pass

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

    @classmethod
    @abstractmethod
    def initialize(cls, **kwargs) -> "BaseLayer":
        pass

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
    def back_prop(self, da: np.ndarray) -> np.ndarray:
        pass

    @property
    def trainable(self) -> bool:
        """
        Tells whether the layer is currently being trained.
        :return:
        """
        return self.__trainable

    def set_trainable(self, trainable: bool = True):
        """
        Sets the layer's trainability.
        :param trainable: True to make the layer trainable.
        :return:
        """
        self.__trainable = trainable

    def freeze_params(self):
        """
        Sets the layer to be non-trainable.
        :return:
        """
        self.__trainable = False


class UnweightedLayer(BaseLayer, ABC):
    """
    Layers with no weights to optimize.
    """

    def set_update_strategy(self, strategy_name: str, **kwargs):
        pass


class WeightedLayer(BaseLayer, ABC):
    """
    Layers with weights to optimize.
    """

    @abstractmethod
    def compute_derivatives(self, da: np.ndarray):
        """
        Compute derivatives by back propagation.
        :param da:
        :return:
        """
        pass

    @abstractmethod
    def _fix_weights(self, *args, **kwargs):
        """
        Set new values to the layer's weights.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def set_weights(self, *args, **kwargs):
        """
        Updates the model's weights only if trainable is set to True.
        :param args:
        :param kwargs:
        :return:
        """
        if self.trainable:
            self._fix_weights(*args, **kwargs)
