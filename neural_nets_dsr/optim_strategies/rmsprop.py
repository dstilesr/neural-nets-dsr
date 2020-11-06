import numpy as np
from .base import UpdateStrategy
from ..utils import ExpAvgAccumulator as ExpAvg


class RMSPropStrategy(UpdateStrategy):
    """
    Update parameters by Root Mean Square Propagation.
    """

    NAME: str = "rms_prop"

    def __init__(
            self,
            learning_rate: float,
            l2_param: float = 0.01,
            beta: float = 0.99,
            normalize: bool = False,
            epsilon: float = 1e-6):

        assert 0.0 < beta < 0.1
        self.__learning_rate = learning_rate
        self.__l2_param = l2_param
        self.__beta = beta
        self.__accumulator: ExpAvg = None
        self.__normalize = normalize
        self.__epsilon = epsilon

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

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Update values by RMS prop.
        :param vals:
        :param grad:
        :return:
        """
        if self.__accumulator is None:
            self.__accumulator = ExpAvg.create(
                vals.shape,
                self.beta,
                self.normalize
            )

        self.__accumulator.update_value(np.square(grad))
        denom = np.sqrt(self.__accumulator.value + self.epsilon)
        full_grad = grad / denom + self.l2_param * vals
        return vals - self.lr * full_grad
