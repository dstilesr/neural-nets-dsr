import numpy as np
from typing import Union, Tuple


class ExpAvgAccumulator:
    """
    Class to compute exponential averages.
    """

    def __init__(
            self,
            initial_val: Union[float, np.ndarray],
            beta: float,
            normalize: bool = False):
        """
        Initializes the accumulator.
        :param initial_val: Initial Value.
        :param beta: Weight factor for the average.
        :param normalize: Normalize to improve precision of early updates.
        """
        self.__value = initial_val
        assert 0. < beta < 1.
        self.__beta = beta
        self._normalize = normalize
        self._counter = 1

    @classmethod
    def create(
            cls,
            input_shape: Tuple,
            beta: float = 0.9,
            normalize: bool = False):
        """
        Initialize accumulator with zeroes.
        :param input_shape: Shape of input array.
        :param beta:
        :param normalize:
        :return: New accumulator.
        """
        return cls(np.zeros(input_shape), beta, normalize)

    @property
    def value(self) -> Union[float, np.ndarray]:
        """
        Value currently stored in the accumulator.
        :return:
        """
        if self._normalize:
            val = self.__value / (1 - self.beta ** self._counter)
        else:
            val = self.__value
        return val

    @property
    def beta(self) -> float:
        """
        Weight factor for the average.
        :return:
        """
        return self.__beta

    def update_value(self, new_val: Union[float, np.ndarray]) -> None:
        """
        Update stored average with the given value.
        :param new_val:
        :return:
        """
        self.__value = (
                self.beta * self.__value
                + (1 - self.beta) * new_val
        )
        if self._normalize:
            self._counter += 1
