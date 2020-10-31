import numpy as np
from typing import Tuple
from .base import BaseLayer
from .numeric_utils import max_pool_2d, avg_pool_2d, expand_pooled


class Base2DPool(BaseLayer):
    """
    Abstract class for 2D pooling layers.
    """

    def __init__(self, filter_x: int = 2, filter_y: int = 2):
        """

        :param filter_x: Height of the filter.
        :param filter_y: Width of the filter.
        """
        assert filter_x > 0 and filter_y > 0
        super().__init__()
        self.__filter_x = filter_x
        self.__filter_y = filter_y
        self.__input_shape = None
        self._cache = {}

    @classmethod
    def initialize(cls, filter_size: Tuple[int, int]) -> "Base2DPool":
        """
        Initialize a pooling layer.
        :param filter_size:
        :return:
        """
        return cls(*filter_size)

    @property
    def filter_shape(self) -> Tuple[int, int]:
        """
        Shape of the pooling filter.
        :return: Two integers.
        """
        return self.__filter_x, self.__filter_y

    def get_output_shape(
            self,
            input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Gives the output shape for a given input shape.
        :param input_shape:
        :return:
        """
        out = (
            input_shape[0],
            input_shape[1] // self.__filter_x,
            input_shape[2] // self.__filter_y,
            input_shape[3]
        )
        return out

    def _fix_weights(self, *args, **kwargs):
        """
        Dummy method for compatibility.
        :param args:
        :param kwargs:
        :return:
        """
        pass


class MaxPool(Base2DPool):
    """
    2D Max pooling for convnets.
    """

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward prop through max pooling layer.
        :param x:
        :param train_mode:
        :return:
        """
        fx, fy = self.filter_shape
        x_pool = max_pool_2d(x, fx, fy)
        if train_mode:
            orig_x, orig_y = x.shape[1:3]
            expanded = expand_pooled(x_pool, orig_x, orig_y, fx, fy)
            self._cache["mask"] = expanded == x
        return x_pool

    def back_prop(self, da: np.ndarray):
        """
        Backprop through max pool layer.
        :param da:
        :return:
        """
        orig_x, orig_y = self._cache["mask"].shape[1:3]
        expanded = expand_pooled(da, orig_x, orig_y, *self.filter_shape)
        daprev = self._cache["mask"] * expanded
        self._cache = {}
        return np.zeros((1, 1)), np.zeros((1, 1)), daprev


class AvgPool(Base2DPool):
    """
    2D Average pooling for convnets.
    """

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward prop through avg pooling layer.
        :param x: Activations of previous layer.
        :param train_mode: Keep cache for backprop.
        :return:
        """
        fx, fy = self.filter_shape
        x_pool = avg_pool_2d(x, fx, fy)

        if train_mode:
            self._cache["a_prev"] = x
        return x_pool

    def back_prop(self, da: np.ndarray):
        """
        Backprop through average pooling layer.
        :param da:
        :return:
        """
        fx, fy = self.filter_shape
        norm_term = 1.0 / (fx * fy)
        orig_x, orig_y = self._cache["a_prev"].shape[1:3]
        daprev = expand_pooled(da, orig_x, orig_y, fx, fy) * norm_term
        return np.zeros((1, 1)), np.zeros((1, 1)), daprev
