import numpy as np
from typing import List, Union
from .networklayer import NetworkLayer
from .activations import ActivationFunc


class NeuralNet:
    """
    Class to represent a Neural Network.
    """

    def __init__(self, layers: List[NetworkLayer]):
        self.layers = layers

    @classmethod
    def initialize(
            cls,
            layer_dims: List[int],
            activations: List[Union[str, ActivationFunc]],
            seed: int = 321):
        """
        Initializes the network.
        :param layer_dims: Dimensions of the network layers. First entry is
            the number of input features.
        :param activations: Activation functions for each layer.
        :param seed: Seed for Numpy RNG.
        :return: The initialized network.
        """
        if (len(layer_dims) + 1) != len(activations):
            raise ValueError("Incompatible lengths!")

        layers = []
        for d in range(1, layer_dims):
            layers.append(NetworkLayer.initialize(
                layer_dims[d],
                layer_dims[d - 1],
                activations[d - 1],
                seed=seed
            ))
            seed += 1

        return cls(layers)

    def compute_predictions(self, x: np.ndarray) -> np.ndarray:
        """
        Computes predictions by applying forward propagation.
        :param x: Array of inputs. Columns represent individual examples.
        :return: Array of predictions.
        """
        out = x
        for layer in self.layers:
            out = layer.forward_prop(x)
        return out


