import numpy as np
from .base import UpdateStrategy
from ..utils import ExpAvgAccumulator as ExpAvg


class AdamStrategy(UpdateStrategy):
    """
    Update parameters with Adaptive Momentum.
    """

    NAME: str = "adam"

    def __init__(
            self,
            learning_rate: float,
            l2_param: float = 0.01,
            beta_momentum: float = 0.9,
            beta_rms: float = 0.99,
            normalize: bool = True,
            epsilon: float = 1e-6):

        assert 0.0 < beta_momentum < 1.0
        self.__learning_rate = learning_rate
        self.__l2_param = l2_param
        self.__beta_mom = beta_momentum
        self.__beta_rms = beta_rms
        self.__accum_rms: ExpAvg = None
        self.__accum_mom: ExpAvg = None
        self.__normalize = normalize
        self.__epsilon = epsilon

    @property
    def lr(self) -> float:
        return self.__learning_rate

    @property
    def l2_param(self) -> float:
        return self.__l2_param

    @property
    def beta_momentum(self) -> float:
        return self.__beta_mom

    @property
    def beta_rms(self) -> float:
        return self.__beta_rms

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
        if self.__accum_mom is None or self.__accum_rms is None:
            self.__accum_rms = ExpAvg.create(
                vals.shape,
                self.beta_rms,
                self.normalize
            )
            self.__accum_mom = ExpAvg.create(
                vals.shape,
                self.beta_momentum,
                self.normalize
            )

        self.__accum_rms.update_value(np.square(grad))
        self.__accum_mom.update_value(grad)

        denom = np.sqrt(self.__accum_rms.value + self.epsilon)
        full_grad = self.__accum_mom.value / denom + self.l2_param * vals
        return vals - self.lr * full_grad

