import numpy as np
from ..network import NeuralNet
from typing import Union, List, Tuple
from ..cost_functions.base import CostFunction
from ..utils import ExpAvgAccumulator as ExpAvg
from .regularized_gradient_descent import GradientDescentL2


class AdamOptimizer(GradientDescentL2):
    """
    Adaptive Momentum (ADAM) Optimizer.
    """

    def __init__(
            self, cost_func: Union[str, CostFunction],
            epochs: int = 600,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            axis: int = 1,
            batch_size: int = 512,
            beta_momentum: float = 0.9,
            beta_rms: float = 0.99,
            epsilon: float = 1e-7,
            verbose: bool = False):
        """

        :param cost_func:
        :param epochs:
        :param learning_rate:
        :param l2_param:
        :param axis:
        :param batch_size:
        :param beta_momentum: Beta for momentum parameter.
        :param beta_rms: Beta for RMS.
        :param epsilon:
        :param verbose:
        """
        assert 0. < beta_momentum < 1., "Invalid beta parameter! Must satisfy 0 < beta < 1."
        assert 0. < beta_rms < 1., "Invalid beta parameter! Must satisfy 0 < beta < 1."
        super().__init__(
            cost_func,
            epochs,
            batch_size,
            axis,
            learning_rate,
            l2_param,
            verbose=verbose
        )
        self._batch_size = batch_size
        self._beta_mom = beta_momentum
        self._beta_rms = beta_rms
        self._momentum_b: List[ExpAvg] = []
        self._rms_b: List[ExpAvg] = []
        self._momentum_w: List[ExpAvg] = []
        self._rms_w: List[ExpAvg] = []
        self._epsilon = epsilon
        self.__iter = 0

    @property
    def beta_momentum(self) -> float:
        return self._beta_mom

    @property
    def beta_rms(self) -> float:
        return self._beta_rms

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def update_momentum_rms(
            self,
            dw: np.ndarray,
            db: np.ndarray,
            lyr_num: int):
        """
        Updates the momentum and RMS prop parameters with the current
        gradients.
        :param dw:
        :param db:
        :param lyr_num:
        :return:
        """
        self._momentum_w[lyr_num].update_value(dw)
        self._momentum_b[lyr_num].update_value(db)
        # RMS Param updates
        self._rms_w[lyr_num].update_value(np.square(dw))
        self._rms_b[lyr_num].update_value(np.square(db))

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
        self.update_momentum_rms(dw, db, lyr_index)
        wreg = self.l2_param * w

        wup = (self.learning_rate * (
            self._momentum_w[lyr_index].value
            / np.sqrt(self.epsilon + self._rms_w[lyr_index].value)
            + wreg
        ))
        bup = (self.learning_rate * (
                self._momentum_b[lyr_index].value
                / np.sqrt(self.epsilon + self._rms_b[lyr_index].value)
        ))
        return w - wup, b - bup

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
            self._momentum_w.append(
                ExpAvg.create(lyr.weights.shape, self.beta_momentum, True)
            )
            self._momentum_b.append(
                ExpAvg.create(lyr.biases.shape, self.beta_momentum, True)
            )
            self._rms_w.append(
                ExpAvg.create(lyr.weights.shape, self.beta_rms, True)
            )
            self._rms_b.append(
                ExpAvg.create(lyr.biases.shape, self.beta_rms, True)
            )

        return super().__call__(network, x, y)
