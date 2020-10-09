import numpy as np
from . import batch_iter
from typing import Union
from .base import Optimizer
from ..network import NeuralNet
from ..cost_functions import CostFunction, COST_NAMES


class GradientDescent(Optimizer):
    """
    Gradient Descent optimizer for neural networks.
    """

    def __init__(
            self,
            cost_func: Union[str, CostFunction],
            epochs: int = 150,
            learning_rate: float = 0.1,
            batch_size: int = -1,
            axis: int = 1,
            shuffle: bool = True,
            verbose: bool = False):
        """

        :param cost_func:
        :param epochs:
        :param learning_rate:
        :param batch_size: Size of minibatches (-1 -> full batch).
        :param axis: Axis along which to split training examples.
        :param shuffle:
        :param verbose: Print cost every 100 iterations if true.
        """
        self._learning_rate = learning_rate
        self._network: NeuralNet = None
        self._verbose = verbose
        self._epochs = epochs
        self._batch_size = batch_size
        self._batch_iter = None
        self._shuffle = shuffle
        self._axis = axis

        if isinstance(cost_func, CostFunction):
            self.__cost = cost_func
        elif cost_func in COST_NAMES.keys():
            self.__cost = COST_NAMES[cost_func]
        else:
            raise ValueError("Unknown cost function!")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cost_func(self) -> CostFunction:
        return self.__cost

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def axis(self) -> int:
        return self._axis

    def make_batch_iterator(
            self,
            x: np.ndarray,
            y: np.ndarray) -> None:
        """
        Makes a batch iterator to run gradient descent.
        :param x:
        :param y:
        :return:
        """
        if self.batch_size <= 0:
            self._batch_iter = batch_iter.FullBatchIterator(
                x,
                y,
                self._axis,
                self._epochs
            )
        else:
            self._batch_iter = batch_iter.MiniBatchIterator(
                x,
                y,
                self._axis,
                self._batch_size,
                self._epochs,
                self._shuffle
            )

    def gradient_descent_iteration(
            self,
            x_batch: np.ndarray,
            y_batch: np.ndarray) -> float:
        """
        Does a single iteration of gradient descent
        :param x_batch: Training set features.
        :param y_batch: Training set labels.
        :return: Cost before the backprop.
        """
        if self._network is None:
            raise NotImplementedError("No network selected!")

        y_pred = self._network.compute_predictions(x_batch, True)
        cost = self.cost_func(y_batch, y_pred)
        da = self.cost_func.gradient(y_batch, y_pred)

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
        self.make_batch_iterator(x, y)
        for i, (x_batch, y_batch) in enumerate(self._batch_iter):
            cost = self.gradient_descent_iteration(x_batch, y_batch)
            if self.verbose and i % 100 == 0:
                print("Cost at iteration %d: %f" % (i, cost))

        return self._network
