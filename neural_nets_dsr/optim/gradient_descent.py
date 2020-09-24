import numpy as np
from typing import Union
from .base import Optimizer
from ..network import NeuralNet
from ..cost_functions import CostFunction, COST_NAMES


class GradientDescent(Optimizer):
    """
    Batch Gradient Descent optimizer for neural networks.
    """

    def __init__(
            self,
            cost_func: Union[str, CostFunction],
            max_iterations: int = 700,
            learning_rate: float = 0.1,
            verbose: bool = False):
        """

        :param cost_func:
        :param max_iterations:
        :param learning_rate:
        :param verbose: Print cost every 100 iterations if true.
        """
        self._learning_rate = learning_rate
        self._network: NeuralNet = None
        self._verbose = verbose
        self._max_iter = max_iterations

        if isinstance(cost_func, CostFunction):
            self.__cost = cost_func
        elif cost_func in COST_NAMES.keys():
            self.__cost = COST_NAMES[cost_func]
        else:
            raise ValueError("Unknown cost function!")

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def cost_func(self) -> CostFunction:
        return self.__cost

    def set_verbose(self, verbose: bool):
        """
        Change verbosity.
        :param verbose:
        :return:
        """
        self._verbose = verbose

    def gradient_descent_iteration(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Does a single iteration of gradient descent
        :param x: Training set features.
        :param y: Training set labels.
        :return: Cost before the backprop.
        """
        if self._network is None:
            raise NotImplementedError("No network selected!")

        y_pred = self._network.compute_predictions(x, True)
        cost = self.cost_func(y, y_pred)
        da = self.cost_func.gradient(y, y_pred)

        for lyr in self._network.layers[::-1]:
            dw, db, da = lyr.back_prop(da)
            lyr.set_weights(
                w=lyr.weights - self.learning_rate * dw,
                b=lyr.biases - self.learning_rate * db
            )
        return cost

    def __call__(
            self,
            network: NeuralNet,
            x: np.ndarray,
            y: np.ndarray) -> NeuralNet:
        """
        Fits the given network by batch gradient descent.
        :param network:
        :param x: Training set features.
        :param y: Training set labels.
        :return: The fitted network.
        """
        self._network = network

        for i in range(self._max_iter):
            cost = self.gradient_descent_iteration(x, y)
            if self.verbose and i % 100 == 0:
                print("Cost at iteration %d: %f" % (i, cost))

        return self._network
