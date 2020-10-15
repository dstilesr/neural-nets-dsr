import numpy as np
from typing import List
from .base import BaseLayer


class DropoutLayer(BaseLayer):
    """
    Dropout regularization layer.
    """

    def __init__(
            self,
            input_shape: List[int],
            dropout_rate: float = 0.2,
            seed: int = 1):
        """

        :param input_shape: Shape of inputs. Examples axis should have a value
            of 1.
        :param dropout_rate:
        :param seed: Seed for RNG.
        """
        if not 0. <= dropout_rate <= 1.:
            raise ValueError("Dropout rate must be between 0 and 1!")

        self.__drop_rate = dropout_rate
        self.__keep_rate = 1. - dropout_rate
        self.__seed = seed
        self.__input_shape = input_shape
        self._mask = None

    @property
    def dropout_rate(self) -> float:
        """
        Rate at which the layer zeroes out connections.
        :return:
        """
        return self.__drop_rate

    @property
    def input_shape(self) -> List[int]:
        """
        Shape of layer inputs. Examples axis must have length 1.
        :return:
        """
        return self.__input_shape

    @property
    def weights(self) -> np.ndarray:
        return np.zeros((1, 1))

    @property
    def biases(self) -> np.ndarray:
        return np.zeros((1, 1))

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward prop through dropout layer.
        :param x:
        :param train_mode:
        :return:
        """
        assert x.ndim == len(self.input_shape), "Incompatible shapes!"
        if train_mode:
            np.random.seed(self.__seed)
            self._mask = np.random.rand(*x.shape) < self.__keep_rate
            out = (x * self._mask) / self.__keep_rate
            self.__seed += sum(x.shape)  # Update seed for RNG
        else:
            out = x
        return out

    def back_prop(self, da: np.ndarray):
        """
        Backprop for dropout layer.
        :param da:
        :return:
        """
        daprev = (self._mask * da) / self.__keep_rate
        self._mask = None
        return np.zeros((1, 1)), np.zeros((1, 1)), daprev

    def set_weights(self, *args, **kwargs):
        """
        Dummy method for compaibility.
        :param args:
        :param kwargs:
        :return:
        """
        pass
