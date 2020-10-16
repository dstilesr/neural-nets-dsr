import numpy as np
from ..network import NeuralNet
from typing import Union, List, Tuple
from ..cost_functions import CostFunction
from ..utils import ExpAvgAccumulator as ExpAvg
from .regularized_gradient_descent import GradientDescentL2


class GradientDescentWithMomentum(GradientDescentL2):
    """
    Mini batch gradient descent with momentum.
    """

    def __init__(
            self, cost_func: Union[str, CostFunction],
            epochs: int = 600,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            batch_size: int = 512,
            beta: float = 0.9,
            axis: int = 1,
            verbose: bool = False):
        """

        :param cost_func: Cost function to optimize.
        :param epochs: Number of full train set passes to perform.
        :param learning_rate:
        :param l2_param: Parameter for L2 regularization.
        :param batch_size: Minibatch size.
        :param beta: Meta parameter for momentum term.
        :param axis:
        :param verbose: Print cost every 100 epochs.
        """
        assert 0. < beta < 1., "Invalid beta parameter! Must satisfy 0 < beta < 1."
        super().__init__(
            cost_func,
            epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2_param=l2_param,
            axis=axis,
            verbose=verbose
        )
        self._batch_size = batch_size
        self._beta = beta
        self._mom_b: List[ExpAvg] = []
        self._mom_w: List[ExpAvg] = []

    @property
    def beta(self) -> float:
        return self._beta

    def get_updates(
            self,
            w: np.ndarray,
            b: np.ndarray,
            dw: np.ndarray,
            db: np.ndarray,
            lyr_index: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param w:
        :param b:
        :param dw:
        :param db:
        :param lyr_index:
        :return:
        """
        self._mom_w[lyr_index].update_value(dw)
        self._mom_b[lyr_index].update_value(db)

        wreg = self.l2_param * w
        wnew = w - self.learning_rate * (self._mom_w[lyr_index].value + wreg)
        bnew = b - self.learning_rate * self._mom_b[lyr_index].value
        return wnew, bnew

    def __call__(
            self,
            network: NeuralNet,
            x: np.ndarray,
            y: np.ndarray) -> NeuralNet:
        """

        :param network:
        :param x:
        :param y:
        :return:
        """
        for lyr in network.layers:
            self._mom_w.append(
                ExpAvg.create(lyr.weights.shape, self.beta)
            )
            self._mom_b.append(
                ExpAvg.create(lyr.biases.shape, self.beta)
            )

        return super().__call__(network, x, y)
