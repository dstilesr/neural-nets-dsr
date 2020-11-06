import numpy as np
from . import cost_functions
from typing import List, Union
from .layers import DenseLayer
from .layers.base import BaseLayer
from .activations import ActivationFunc
from .utils import FullBatchIterator, MiniBatchIterator

U = Union[str, cost_functions.CostFunction]


class NeuralNet:
    """
    Class to represent a Neural Network.
    """

    def __init__(self, layers: List[BaseLayer]):
        self.layers = layers
        self._batch_iter = None

    @staticmethod
    def get_cost_func(cost_name: U) -> cost_functions.CostFunction:
        """
        Gets a cost function from its name.
        :param cost_name:
        :return:
        """
        if isinstance(cost_name, cost_functions.CostFunction):
            out = cost_name
        elif cost_name in cost_functions.AVAILABLE_COST_FUNCS:
            out = cost_functions.COST_NAMES[cost_name]
        else:
            raise ValueError("Unknown cost function!")
        return out

    @property
    def depth(self) -> int:
        """
        Number of layers in the network (not including input).
        :return:
        """
        return len(self.layers)

    @classmethod
    def initialize(
            cls,
            layer_dims: List[int],
            activations: List[Union[str, ActivationFunc]],
            xavier: bool = False,
            seed: int = 321):
        """
        Initializes the network with dense, fully connected layers.
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
            train_mode: bool = False) -> np.ndarray:
        """
        Computes predictions by applying forward propagation.
        :param x: Array of inputs. Columns represent individual examples.
        :param train_mode: Keep activation caches in layers for backprop.
        :return: Array of predictions.
        """
        out = x
        for layer in self.layers:
            out = layer.forward_prop(out, train_mode=train_mode)
        return out

    def layer_types(self) -> dict:
        """
        Gives a dictionary with the type of each layer in the network.
        :return:
        """
        out = {}
        for i, lyr in enumerate(self.layers):
            out[f"Layer_{i:02d}"] = str(type(lyr))

        return out

    def make_batch_iterator(
            self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            batch_size: int) -> None:
        """
        Makes a batch iterator to run gradient descent.
        :param x:
        :param y:
        :param epochs:
        :param batch_size:
        :return:
        """
        axis = 0 if x.ndim == 4 else 1
        if batch_size <= 0:
            self._batch_iter = FullBatchIterator(
                x,
                y,
                axis,
                epochs
            )
        else:
            self._batch_iter = MiniBatchIterator(
                x,
                y,
                axis,
                batch_size,
                epochs,
                True
            )

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cost_function: U,
            optim_strategy: str = "gradient_descent",
            epochs: int = 100,
            batch_size: int = -1,
            verbose: bool = False,
            print_interval: int = 100,
            **kwargs) -> "NeuralNet":
        """
        Fits the network to the given data.
        :param x: Training set features.
        :param y: Training set labels.
        :param cost_function: Cost funtion to use in training.
        :param optim_strategy: Which optimizer to use.
        :param epochs: Number of training epochs.
        :param batch_size: Size of batch to use in training.
        :param verbose: Periodically print cost.
        :param print_interval: Number of iterations between cost prints.
        :param kwargs: Hyperparameters for optimizer.
        :return: The fitted network.
        """
        for lyr in self.layers:
            lyr.set_update_strategy(optim_strategy, **kwargs)

        cost = self.get_cost_func(cost_function)
        self.make_batch_iterator(x, y, epochs, batch_size)
        iter_count = 0

        for batch_x, batch_y in self._batch_iter:
            iter_count += 1
            preds = self.compute_predictions(batch_x, train_mode=True)
            da = cost.gradient(batch_y, preds)
            for lyr in self.layers[::-1]:
                da = lyr.back_prop(da)

            if verbose and iter_count % print_interval == 0:
                cost_val = cost(batch_y, preds)
                print("Cost at iteration %03d: %0.5f" % (
                    iter_count,
                    cost_val
                ))

        if verbose:
            all_pred = self.compute_predictions(x)
            total_cost = cost(y, all_pred)
            print("Training ended! Final cost: %0.5f" % total_cost)
        return self


