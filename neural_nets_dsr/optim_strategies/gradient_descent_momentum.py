import numpy as np
from .base import UpdateStrategy
from ..utils import ExpAvgAccumulator as ExpAvg


class GDMomentumStrategy(UpdateStrategy):
    """
    Update parameters via gradient descent with momentum.
    """

    NAME: str = "gradient_descent_momentum"

    def __init__(
            self,
            learning_rate: float,
            l2_param: float = 0.01,
            beta: float = 0.9,
            normalize: bool = False):

        assert 0.0 < beta < 1.0
        self.__learning_rate = learning_rate
        self.__l2_param = l2_param
        self.__beta = beta
        self.__accumulator: ExpAvg = None
        self.__normalize = normalize

    @property
    def lr(self) -> float:
        return self.__learning_rate

    @property
    def l2_param(self) -> float:
        return self.__l2_param

    @property
    def beta(self) -> float:
        return self.__beta

    @property
    def normalize(self) -> bool:
        return self.__normalize

    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Update parameters using Gradient Descent with Momentum.
        :param vals:
        :param grad:
        :return:
        """
        assert vals.shape == grad.shape
        if self.__accumulator is None:
            self.__accumulator = ExpAvg.create(
                vals.shape,
                self.beta,
                self.normalize
            )

        self.__accumulator.update_value(grad)
        full_grad = self.__accumulator.value + self.l2_param * grad
        return vals - self.lr * full_grad
