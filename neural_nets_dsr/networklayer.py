import numpy as np
from typing import Union
from . import activations


class NetworkLayer:
    """
    Class to represent a layer of a neural network.
    """

    def __init__(
            self,
            w: np.ndarray,
            b: np.ndarray,
            activation: activations.ActivationFunc):

        self.w = w
        self.b = b
        self.__activation = activation
        self.cache = {}

    @classmethod
    def initialize(
            cls,
            num_neurons: int,
            prev_layer_neurons: int,
            activation: Union[str, activations.ActivationFunc] = "relu",
            seed=123,
            scale: float = 0.01):
        """
        Initializes the parameters for the layer with random weights.
        :param num_neurons: Number of neurons in this layer.
        :param prev_layer_neurons: Number of neurons in previous layer.
        :param activation: Activation function.
        :param seed: Seed for numpy RNG.
        :param scale: Scale factor for initial weights.
        :return:
        """
        if isinstance(activation, activations.ActivationFunc):
            act = activation
        elif activation in activations.ACTIVATIONS_NAMES.keys():
            act = activations.ACTIVATIONS_NAMES[activation]
        else:
            raise ValueError("Unknown activation function!")

        np.random.seed(seed)
        obj = cls(
            w=np.random.randn(num_neurons, prev_layer_neurons) * scale,
            b=np.zeros((num_neurons, 1)),
            activation=act
        )
        return obj

    @property
    def activation(self) -> activations.ActivationFunc:
        """
        Read-only activation function for this layer.
        :return:
        """
        return self.__activation

    def reset_cache(self):
        """
        Deletes the cached value of the forward prop.
        :return:
        """
        self.cache = {}

    def forward_prop(
            self,
            x: np.ndarray,
            keep_cache: bool = False) -> np.ndarray:
        """
        Computes the forward propagation of the layer.
        :param x: Array of activations from previous layer. Columns correspond
            to different examples, rows to different features.
        :param keep_cache: Keep a cache of the activation for later backprop
            or not.
        :return:
        """
        z = np.dot(self.w, x) + self.b
        if keep_cache:
            self.cache["z"] = z
            self.cache["a"] = x
        return self.activation(z)

    def back_prop(self, da: np.ndarray):
        """
        Computes the derivatives of the cost function with respect to the
        weights and biases of this layer using back propagation.
        :param da: Derivative of the cost function with respect to the
            activations of this layer.
        :return: The gradients dW, db, da (for previous layer)
        """
        z = self.cache["z"]
        m = z.shape[1]
        dz = da * self.activation.gradient(z)
        dw = np.dot(dz, self.cache["a"].T) / m
        db = np.mean(dz, axis=1, keepdims=True)

        self.reset_cache()
        return dw, db, np.dot(self.w.T, dz)
