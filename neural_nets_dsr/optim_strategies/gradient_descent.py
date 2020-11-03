import numpy as np
from . import batch_iter
from .base import Optimizer, UpdateStrategy
from typing import Union, Tuple
from ..network import NeuralNet
from ..cost_functions.base import CostFunction


class GradientDescentStrategy(UpdateStrategy):
    """
    Compute updates by gradient descent.
    """

    def __init__(self, learning_rate: float):
        assert learning_rate > 0.0
        self.__learning_rate = learning_rate

    @property
    def lr(self) -> float:
        return self.__learning_rate

    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Updates parameters by gradient descent.
        :param vals:
        :param grad:
        :return:
        """
        assert vals.shape == grad.shape
        return vals - self.lr * grad


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

        self.__cost = self.get_cost_func(cost_func)
        self.__print_interval = 100

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

    def set_print_interval(self, interval: int = 100):
        """
        Sets interval (number of iterations) between prints when training and
        verbose.
        :param interval: Positive integer.
        :return:
        """
        assert interval > 0
        self.__print_interval = int(interval)

    def local_print(self, *args, **kwargs):
        """
        Prints arguments only if verbose is set to True.
        :param args:
        :param kwargs:
        :return:
        """
        if self._verbose:
            print(*args, **kwargs)

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

    def get_updates(
            self,
            w: np.ndarray,
            b: np.ndarray,
            dw: np.ndarray,
            db: np.ndarray,
            lyr_index: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient descent updated weights and biases.
        :param w: Weights of a layer.
        :param b: Biases of a layer
        :param dw: Gradient of cost wrt weights.
        :param db: Gradient of cost wrt biases.
        :param lyr_index: Index of layer within network.
        :return: The updated weights and biases.
        """
        return w - self.learning_rate * dw, b - self.learning_rate * db

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

        for i in reversed(range(self._network.depth)):
            lyr = self._network.layers[i]
            dw, db, da = lyr.back_prop(da)
            wnew, bnew = self.get_updates(lyr.weights, lyr.biases, dw, db, i)
            self._network.layers[i].set_weights(w=wnew, b=bnew)
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
        cost = -1.0
        self._network = network
        self.make_batch_iterator(x, y)
        for i, (x_batch, y_batch) in enumerate(self._batch_iter):
            cost = self.gradient_descent_iteration(x_batch, y_batch)
            if i % self.__print_interval == 0:
                self.local_print("Cost at iteration %03d: %0.4f" % (i, cost))

        self.local_print("Training ended. Cost: %0.4f" % cost)

        return self._network
