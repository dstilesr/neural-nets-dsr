import numpy as np
from typing import Union, Callable
T = Union[np.ndarray, float]


class ActivationFunc(Callable[[T], T]):
    """
    Base class for activation functions.
    """

    def __init__(
            self,
            function: Callable[[T], T],
            gradient: Callable[[T], T]):
        """

        :param function: Activation function.
        :param gradient: Derivative of activation function.
        """
        self.__function = function
        self.__gradient = gradient

    @property
    def gradient(self) -> Callable[[T], T]:
        """
        Gradient function (derivative) of the activation (read only).
        :return:
        """
        return self.__gradient

    def __call__(self, inputs: T) -> T:
        """
        Calls the function on the given inputs.
        :param inputs: Float or NumPy array.
        :return: Float or NumPy array.
        """
        return self.__function(inputs)

