import numpy as np
from .base import WeightedLayer
from typing import Union, Optional
from ..activations.base import ActivationFunc
from ..optim_strategies.base import UpdateStrategy


class BasicRNN(WeightedLayer):
    """
    Simple Recurring Unit.
    """

    def __init__(
            self,
            weight_matrix: np.ndarray,
            biases: np.ndarray,
            activation: ActivationFunc):
        super().__init__()
        self.__weights = weight_matrix
        self.__biases = biases
        self.__activation = activation

        self.__hidden_state: int = biases.shape[0]

        self.__w_update: Optional[UpdateStrategy] = None
        self.__b_update: Optional[UpdateStrategy] = None

        self.__cache = {}

    @classmethod
    def initialize(
            cls,
            num_neurons: int,
            input_size: int,
            activation: Union[str, ActivationFunc] = "tanh",
            seed: int = 123,
            scale: float = 0.01) -> "BasicRNN":
        """
        Initialize the Recurring network layer.
        :param num_neurons: Size of hidden state.
        :param input_size: Dimension of input vectors.
        :param activation: Activation function to use.
        :param seed: Seed for RNG.
        :param scale: Scaling factor for param initialization.
        :return: Initialized layer.
        """
        assert num_neurons > 0 and input_size > 0
        activation_func = cls.get_activation(activation)
        np.random.seed(seed)
        weights = scale * np.random.randn(
            num_neurons,
            num_neurons + input_size
        )
        biases = scale * np.random.randn(num_neurons, 1)
        return cls(weights, biases, activation_func)

    @property
    def weights(self) -> np.ndarray:
        """
        Weights of this layer.
        :return:
        """
        return self.__weights

    @property
    def biases(self) -> np.ndarray:
        """
        Biases of this layer.
        :return:
        """
        return self.__biases

    @property
    def activation(self) -> ActivationFunc:
        """
        Activation function.
        :return:
        """
        return self.__activation

    @property
    def hidden_state(self) -> int:
        """
        Size of the layer's hidden state vector.
        :return:
        """
        return self.__hidden_state

    def set_update_strategy(self, strategy_name: str, **kwargs):
        """
        Sets the update strategy for the layer parameters.
        :param strategy_name:
        :param kwargs:
        :return:
        """
        update_type = self.strategy_from_name(strategy_name)
        self.__w_update = update_type(**kwargs)
        self.__b_update = update_type(**kwargs)

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward propagation through the recurring layer.
        :param x: Input array. Must be of shape (num_examples,
            sequence_length, input_size).
        :param train_mode: Keep cache for back prop.
        :return:
        """
        a = np.zeros((x.shape[0], self.hidden_state))
        for i in range(x.shape[1]):
            xa = np.hstack((a, x[:, i, :]))
            z = (np.dot(self.weights, xa.T) + self.biases).T
            a = self.activation(z)
        return a

    def _fix_weights(self, *args, **kwargs):
        raise NotImplementedError("WIP")

    def compute_derivatives(self, da: np.ndarray):
        raise NotImplementedError("Backprop WIP")

    def back_prop(self, da: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Backprop WIP")
