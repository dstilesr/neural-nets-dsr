import numpy as np
from typing import Union
from .. import activations
from ..optim_strategies.base import UpdateStrategy
from .base import WeightedLayer


class DenseLayer(WeightedLayer):
    """
    Class to represent a layer of a neural network.
    """

    def __init__(
            self,
            w: np.ndarray,
            b: np.ndarray,
            activation: activations.ActivationFunc):
        super().__init__()
        if w.shape[0] != b.shape[0]:
            raise ValueError("Mismatched dimensions!")

        self.__w = w
        self.__b = b
        self.__activation = activation
        self.cache = {}

        self.__b_update: UpdateStrategy = None
        self.__w_update: UpdateStrategy = None

    @classmethod
    def initialize(
            cls,
            num_neurons: int,
            prev_layer_neurons: int,
            activation: Union[str, activations.ActivationFunc] = "relu",
            seed=123,
            scale: float = 0.01) -> "DenseLayer":
        """
        Initializes the parameters for the layer with random weights.
        :param num_neurons: Number of neurons in this layer.
        :param prev_layer_neurons: Number of neurons in previous layer.
        :param activation: Activation function.
        :param seed: Seed for numpy RNG.
        :param scale: Scale factor for initial weights.
        :return:
        """
        act = cls.get_activation(activation)
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

    @property
    def weights(self) -> np.ndarray:
        """
        Read-only weight matrix of the layer.
        :return:
        """
        return self.__w

    @property
    def biases(self) -> np.ndarray:
        """
        Read-only bias vector of the layer
        :return:
        """
        return self.__b

    def reset_cache(self):
        """
        Deletes the cached value of the forward prop.
        :return:
        """
        self.cache = {}

    def set_update_strategy(self, strategy_name: str, **kwargs):
        """
        Sets optimization strategy for weights and biases.
        :param strategy_name:
        :param kwargs: Optimization hyperparameters.
        :return:
        """
        optim_type = self.strategy_from_name(strategy_name)
        self.__w_update = optim_type(**kwargs)
        self.__b_update = optim_type(**kwargs)

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Computes the forward propagation of the layer.
        :param x: Array of activations from previous layer. Columns correspond
            to different examples, rows to different features.
        :param train_mode: Keep a cache of the activation for later backprop
            or not.
        :return:
        """
        z = np.dot(self.__w, x) + self.__b
        if train_mode:
            self.cache["z"] = z
            self.cache["a"] = x
        return self.activation(z)

    def compute_derivatives(self, da: np.ndarray):
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
        return dw, db, np.dot(self.__w.T, dz)

    def _fix_weights(self, w: np.ndarray, b: np.ndarray):
        """
        Updates the weights and biases with new values.
        :param w: Updated weights.
        :param b: Updated biases.
        :return:
        """
        assert w.shape == self.__w.shape and b.shape == self.__b.shape
        self.__w = w
        self.__b = b

    def back_prop(self, da: np.ndarray) -> np.ndarray:
        """
        Backprop through the layer.
        :param da:
        :return: Gradients wrt previous layer's activations.
        """
        if self.__b_update is None or self.__w_update is None:
            raise ValueError("No optimization strategy set!")

        dw, db, daprev = self.compute_derivatives(da)
        new_w = self.__w_update.update_params(self.__w, dw)
        new_b = self.__b_update.update_params(self.__b, db)
        self.set_weights(new_w, new_b)
        return daprev

