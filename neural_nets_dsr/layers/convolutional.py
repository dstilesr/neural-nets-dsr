import numpy as np
from scipy import signal
from typing import Tuple, Union
from .base import BaseLayer, ActivationFunc


class Convolution2D(BaseLayer):
    """
    2D Convolutional Layer.
    """

    def __init__(
            self,
            activation: ActivationFunc,
            filters: np.ndarray,
            biases: np.ndarray,
            stride: int,
            padding: str = "valid"):
        """

        :param filters:
        :param biases:
        :param stride:
        """
        self.__filters = filters
        self.__biases = biases
        self.__stride = stride
        self.__activation = activation
        self.__padding = padding
        self.__filter_size = self.__filters.shape[:2]
        self._cache = {}

    @property
    def filters(self) -> np.ndarray:
        return self.__filters

    @property
    def biases(self) -> np.ndarray:
        return self.__biases

    @property
    def stride(self) -> int:
        return self.__stride

    @property
    def activation(self) -> ActivationFunc:
        return self.__activation

    @property
    def padding(self) -> str:
        return self.__padding

    @classmethod
    def initialize(
            cls,
            prev_channels: int,
            num_filters: int,
            filter_size: Tuple[int, int] = (3, 3),
            stride: int = 1,
            padding: str = "valid",
            activation: Union[str, ActivationFunc] = "relu",
            seed: int = 21):
        """
        Initialize the layer.
        :param prev_channels: Number of channels in previous layer.
        :param num_filters: Number of filters.
        :param filter_size: Size of filters.
        :param stride: NOT CURRENTLY IN USE.
        :param padding: Padding strategy.
        :param activation:
        :param seed: Seed for RNG
        :return:
        """
        np.random.seed(seed)
        filters = np.random.randn(
            *filter_size,
            prev_channels,
            num_filters
        ) * (1. / num_filters)
        biases = np.zeros((1, 1, num_filters))

        out = cls(
            activation=cls.get_activation(activation),
            filters=filters,
            biases=biases,
            stride=stride,
            padding=padding
        )
        return out

    def output_shape(
            self,
            x_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Given the shape of an input array, returns the shape the output will
        have.
        :param x_shape:
        :return:
        """
        if self.padding == "valid":
            output_shape = (
                x_shape[0],
                x_shape[1] - 2 * (self.filters.shape[0] // 2),
                x_shape[2] - 2 * (self.filters.shape[1] // 2),
                self.filters.shape[-1]
            )
        else:
            output_shape = (
                x_shape[0],
                x_shape[1],
                x_shape[2],
                self.filters.shape[-1]
            )
        return output_shape

    def same_pad(self, x: np.ndarray) -> np.ndarray:
        """
        Performs 'same' padding on the input array.
        :param x:
        :return:
        """
        pad_x = self.filters.shape[0] // 2
        pad_y = self.filters.shape[1] // 2
        x_pad = np.pad(x, (
            (0, 0),
            (pad_x, pad_x),
            (pad_y, pad_y),
            (0, 0)
        ), mode="constant", constant_values=(0., 0.))
        return x_pad

    def convolve_filter(self, filter_index: int, x: np.ndarray) -> np.ndarray:
        """
        Convolves a given filter with the input.
        :param filter_index: Index of filter to convolve.
        :param x: Array of shape (num_examples, height, width, channels).
        :return:
        """
        output_shape = self.output_shape(x.shape)[:-1]
        out = np.zeros(output_shape)

        for ex in range(x.shape[0]):
            for i in range(x.shape[-1]):
                out[ex, :, :] += signal.correlate2d(
                    x[ex, :, :, i],
                    self.filters[:, :, i, filter_index],
                    mode=self.padding
                )
        return out

    def forward_prop(
            self,
            x: np.ndarray,
            keep_cache: bool = False) -> np.ndarray:
        """
        Forward propagation on this layer.
        :param x:
        :param keep_cache:
        :return:
        """
        if keep_cache:
            self._cache["a_prev"] = x
            if self.padding == "same":
                self._cache["a_prev_pad"] = self.same_pad(x)
            else:
                self._cache["a_prev_pad"] = x

        z = np.zeros(self.output_shape(x.shape))
        for f in range(self.filters.shape[-1]):
            z[:, :, :, f] = self.convolve_filter(f, x)

        z += self.biases
        return self.activation(z)

    def back_prop(self, da: np.ndarray):
        """

        :param da:
        :return: Gradients of filters and biases and wrt to previous
            activations.
        """
        # TODO UNFINISHED
        raise NotImplementedError()
        dz = self.activation.gradient(da)
        db = np.sum(dz, axis=(1, 2), keepdims=True)
        db = np.mean(db, axis=0, keepdims=False)

        dw = np.zeros(self.filters.shape)
        return dw, db, da

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        """
        Updates filters and biases.
        :param w:
        :param b:
        :return:
        """
        assert w.shape == self.__filters.shape
        self.__filters = w

        assert b.shape == self.__biases.shape
        self.__biases = b


class FlattenLayer(BaseLayer):
    """
    Layer to flatten the output of a convolutional network.
    """

    def __init__(self):
        self._input_shape = None

    def forward_prop(
            self,
            x: np.ndarray,
            keep_cache: bool = False) -> np.ndarray:
        """
        Flattens the output of a convolutional layer into the shape
        (num_features, num_examples).
        :param x: Output of convolutional layer.
        :param keep_cache:
        :return:
        """
        self._input_shape = x.shape
        out_shape = (x.shape[0], np.prod(x.shape[1:], dtype="int"))
        return x.reshape(out_shape).T

    def back_prop(self, da: np.ndarray):
        """

        :param da:
        :return:
        """
        return None, None, da.T.reshape(self._input_shape)

    def set_weights(self, *args, **kwargs):
        pass
