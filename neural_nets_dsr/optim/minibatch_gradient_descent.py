import numpy as np
from typing import Union
from ..network import NeuralNet
from ..cost_functions import CostFunction
from .regularized_gradient_descent import GradientDescentL2


class MiniBatchGDL2(GradientDescentL2):
    """
    L2 regularized gradient descent by mini batches.
    """

    def __init__(
            self, cost_func: Union[str, CostFunction],
            epochs: int = 600,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            batch_size: int = 512,
            verbose: bool = False):
        """

        :param cost_func:
        :param epochs:
        :param learning_rate:
        :param l2_param:
        :param batch_size:
        """
        super().__init__(
            cost_func,
            epochs,
            learning_rate,
            l2_param,
            verbose=verbose
        )
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """
        Read only minibatch size.
        :return:
        """
        return self._batch_size

    def __call__(
            self,
            network: NeuralNet,
            x: np.ndarray,
            y: np.ndarray) -> NeuralNet:
        """
        Fits the given network by mini-batch gradient descent.
        :param network:
        :param x:
        :param y:
        :return:
        """
        self._network = network
        batch_num = x.shape[1] // self.batch_size
        remainder = (x.shape[1] % self.batch_size) == 0

        for e in range(self._max_iter):
            for b in range(batch_num):
                start = b * self.batch_size
                x_mini = x[:, start:(start + self.batch_size)]
                y_mini = y[:, start:(start + self.batch_size)]
                _ = self.gradient_descent_iteration(x_mini, y_mini)

            if remainder:
                start = batch_num * self.batch_size
                _ = self.gradient_descent_iteration(x[start:], y[start:])

            if self.verbose and e % 100 == 0:
                cost = self.cost_func(y, self._network.compute_predictions(x))
                print("Cost on epoch %d: %0.4f" % (e, cost))

        return self._network
