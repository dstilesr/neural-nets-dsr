import numpy as np
from typing import Tuple, Union
from .base import BaseLayer, ActivationFunc
from .numeric_utils import full_conv, conv_backprop


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
        super().__init__()
        self.__filters = filters
        self.__biases = biases
        self.__stride = stride
        self.__activation = activation
        self.__padding = padding
        self.__filter_size = self.__filters.shape[:2]
        self._cache = {}

    @property
    def filters(self) -> np.ndarray:
        """
        Read-only filters of the layer.
        :return:
        """
        return self.__filters

    @property
    def weights(self) -> np.ndarray:
        """
        Alias for filters property.
        :return:
        """
        return self.filters

    @property
    def biases(self) -> np.ndarray:
        """
        Read-only biases for the layer.
        :return:
        """
        return self.__biases

    @property
    def stride(self) -> int:
        return self.__stride

    @property
    def activation(self) -> ActivationFunc:
        """
        Layer's activation function.
        :return:
        """
        return self.__activation

    @property
    def padding(self) -> str:
        """
        Layer's padding strategy.
        :return:
        """
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
            seed: int = 21) -> "Convolution2D":
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
                int(x_shape[1] - 2 * (self.filters.shape[0] // 2)),
                int(x_shape[2] - 2 * (self.filters.shape[1] // 2)),
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

    def pad(self, x: np.ndarray) -> np.ndarray:
        """
        Performs padding on the input array.
        :param x:
        :return:
        """
        if self.padding == "same":
            pad_x = self.filters.shape[0] // 2
            pad_y = self.filters.shape[1] // 2
            x_pad = np.pad(x, (
                (0, 0),
                (pad_x, pad_x),
                (pad_y, pad_y),
                (0, 0)
            ), mode="constant", constant_values=(0., 0.))
        else:
            x_pad = x
        return x_pad

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward propagation on this layer.
        :param x:
        :param train_mode:
        :return:
        """
        xpad = self.pad(x)
        z = full_conv(xpad, self.filters)

        z += self.biases
        a = self.activation(z)

        if train_mode:
            self._cache["a_prev"] = x
            self._cache["z"] = z
            self._cache["a_prev_pad"] = xpad

        return a

    def __compute_dwda(self, dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient of the cost function wrt to the layer's weights
        and previous layer's activations.
        :param dz:
        :return:
        """
        filt_h, filt_w = self.filters.shape[:2]
        dw, da_prev_pd = conv_backprop(
            dz,
            self.filters,
            self._cache["a_prev_pad"]
        )

        if self.padding == "same":
            pdx, pdy = filt_h // 2, filt_w // 2
            da_prev = da_prev_pd[:, pdx:-pdx, pdy:-pdy, :]
        else:
            da_prev = da_prev_pd

        return dw / dz.shape[0], da_prev

    def back_prop(self, da: np.ndarray):
        """
        Back propagation on this layer.
        :param da:
        :return: Gradients wrt filters, biases and to previous layer's
            activations.
        """
        dz = self.activation.gradient(self._cache["z"]) * da
        db = np.sum(dz, axis=(1, 2), keepdims=True)
        db = np.mean(db, axis=0, keepdims=False)
        dw, da_prev = self.__compute_dwda(dz)
        self._cache = {}
        return dw, db, da_prev

    def _fix_weights(self, w: np.ndarray, b: np.ndarray):
        """
        Updates filters and biases.
        :param w:
        :param b:
        :return:
        """
        assert w.shape == self.__filters.shape
        assert b.shape == self.__biases.shape
        self.__filters = w
        self.__biases = b


class FlattenLayer(BaseLayer):
    """
    Layer to flatten the output of a convolutional network.
    """

    def __init__(self):
        super().__init__()
        self._input_shape = None

    @classmethod
    def initialize(cls) -> "FlattenLayer":
        return cls()

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Flattens the output of a convolutional layer into the shape
        (num_features, num_examples).
        :param x: Output of convolutional layer.
        :param train_mode:
        :return:
        """
        self._input_shape = x.shape
        out_shape = (x.shape[0], np.prod(x.shape[1:], dtype="int"))
        return x.reshape(out_shape).T

    def back_prop(self, da: np.ndarray):
        """
        DUMMY
        :param da:
        :return:
        """
        return 0.0, 0.0, da.T.reshape(self._input_shape)

    def _fix_weights(self, *args, **kwargs):
        """
        DUMMY
        :param args:
        :param kwargs:
        :return:
        """
        pass
