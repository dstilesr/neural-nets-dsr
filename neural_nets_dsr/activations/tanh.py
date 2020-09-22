import numpy as np
from .base import ActivationFunc, T


def tanh(x: T) -> T:
    """
    Tanh function.
    :param x:
    :return:
    """
    return np.tanh(x)


def tanh_derivative(x: T) -> T:
    """

    :param x:
    :return:
    """
    th = tanh(x)
    return 1. - np.square(th)


tanh_activation = ActivationFunc(tanh, tanh_derivative)
