import numpy as np
from typing import Tuple
from .base import BaseLayer
from ..utils import ExpAvgAccumulator as ExpAvg


class BatchNorm(BaseLayer):
    """
    Batch Normalization layer.
    """

    def __init__(
            self,
            gamma: np.ndarray,
            beta_shift: np.ndarray,
            beta_avg: float = 0.9,
            epsilon: float = 1e-6,
            axis: int = 1):
        """

        :param gamma:
        :param beta_shift:
        :param beta_avg:
        :param epsilon:
        :param axis:
        """
        super().__init__()
        self._epsilon = epsilon
        self._axis = axis
        self.__gamma = gamma
        self.__beta = beta_shift

        self._mu_accum = ExpAvg.create(beta_shift.shape, beta_avg)
        self._sigma_accum = ExpAvg.create(beta_shift.shape, beta_avg)
        self.__cache = {}

    @classmethod
    def initialize(
            cls,
            input_shape: Tuple,
            beta_avg: float = 0.9,
            epsilon: float = 1e-6,
            axis: int = 1,
            seed: int = 843) -> "BatchNorm":
        """

        :param input_shape: Shape of previous layer's activations (Set
            examples axis to 1!).
        :param beta_avg: Weighting parameter for computing averages.
        :param epsilon: Parameter for numerical stability.
        :param axis: Axis along which different examples are placed.
        :param seed:
        """
        assert input_shape[axis] == 1

        np.random.seed(seed)
        gamma = np.random.randn(*input_shape) * 0.01
        beta_shift = np.random.randn(*input_shape) * 0.01
        return cls(gamma, beta_shift, beta_avg, epsilon, axis)

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def weights(self) -> np.ndarray:
        """
        Scaling (gamma) weights.
        :return:
        """
        return self.__gamma

    @property
    def biases(self) -> np.ndarray:
        """
        Shift (beta) weights.
        :return:
        """
        return self.__beta

    def forward_prop(
            self,
            x: np.ndarray,
            train_mode: bool = False) -> np.ndarray:
        """
        Forward-propagates batch normalization.
        :param x:
        :param train_mode:
        :return:
        """

        if train_mode:
            mu = np.mean(x, axis=self.axis, keepdims=True)
            sigma_sq = np.var(x, axis=self.axis, keepdims=True)
            self._mu_accum.update_value(mu)
            self._sigma_accum.update_value(sigma_sq)

            xnorm = (x - mu) / np.sqrt(sigma_sq + self._epsilon)
            self.__cache["anorm"] = xnorm
            self.__cache["aprev"] = x
            self.__cache["mu"] = mu
            self.__cache["sigma"] = sigma_sq
        else:
            mu = self._mu_accum.value
            sigma_sq = self._sigma_accum.value
            xnorm = (x - mu) / np.sqrt(sigma_sq + self._epsilon)

        return self.__gamma * xnorm + self.__beta

    def back_prop(self, da: np.ndarray):
        """
        Compute gradients of gamma and beta.
        :param da:
        :return:
        """
        dbeta = np.mean(da, axis=self.axis, keepdims=True)
        dgamma = np.mean(da * self.__cache["anorm"], axis=self.axis, keepdims=True)
        m = da.shape[self.axis]  # Number of training examples.
        da_norm = da * self.__gamma

        # Denominator: (x - mu)/sqrt(var + epsilon)
        denom = np.sqrt(self.__cache["sigma"] + self._epsilon)

        # Compute dmu / dx
        dmux = self.__cache["aprev"] / m

        # Compute dvar / dx
        dsigmax = (2. / m) * (self.__cache["aprev"] - dmux)

        # d((x - mu)/sqrt(var + epsilon)) / dx
        da_prev = (
            (1. - dmux) / denom
            + (self.__cache["aprev"] - self.__cache["mu"]) * dsigmax
            / (2. * denom ** 3)
        ) * da_norm
        self.__cache = {}
        return dgamma, dbeta, da_prev

    def _fix_weights(self, w: np.ndarray, b: np.ndarray):
        """
        Sets new values for gamma and beta vectors.
        :param w:
        :param b:
        :return:
        """
        assert w.shape == self.__gamma.shape
        self.__gamma = w

        assert b.shape == self.__beta.shape
        self.__beta = b
