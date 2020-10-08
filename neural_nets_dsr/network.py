import numpy as np
from typing import List, Union
from .layers import DenseLayer
from .activations import ActivationFunc


class NeuralNet:
    """
    Class to represent a Neural Network.
    """

    def __init__(self, layers: List[DenseLayer]):
        self.layers = layers

    @classmethod
    def initialize(
            cls,
            layer_dims: List[int],
            activations: List[Union[str, ActivationFunc]],
            xavier: bool = False,
            seed: int = 321):
        """
        Initializes the network.
        :param layer_dims: Dimensions of the network layers. First entry is
            the number of input features.
        :param activations: Activation functions for each layer.
        :param xavier: Use xavier normalization for weights.
        :param seed: Seed for Numpy RNG.
        :return: The initialized network.
        """
        if (len(layer_dims) - 1) != len(activations):
            raise ValueError("Incompatible lengths!")

        layers = []
        for d in range(1, len(layer_dims)):
            if xavier:
                layers.append(DenseLayer.initialize(
                    layer_dims[d],
                    layer_dims[d - 1],
                    activations[d - 1],
                    scale=np.sqrt(2 / layer_dims[d - 1]),
                    seed=seed
                ))
            else:
                layers.append(DenseLayer.initialize(
                    layer_dims[d],
                    layer_dims[d - 1],
                    activations[d - 1],
                    seed=seed
                ))
            seed += 1

        return cls(layers)

    def compute_predictions(
            self,
            x: np.ndarray,
            keep_caches: bool = False) -> np.ndarray:
        """
        Computes predictions by applying forward propagation.
        :param x: Array of inputs. Columns represent individual examples.
        :param keep_caches: Keep activation caches in layers for backprop.
        :return: Array of predictions.
        """
        out = x
        for layer in self.layers:
            out = layer.forward_prop(out, keep_cache=keep_caches)
        return out

    def append(self, other):
        """
        Appends the layers of the other network.
        :param other: Another NeuralNet.
        :return: None
        """
        fst_weights = other.layers[0].weights.shape[1]
        if fst_weights != self.layers[-1].weights.shape[0]:
            raise ValueError("Incompatible Lengths!")

        self.layers.append(other.layers)


