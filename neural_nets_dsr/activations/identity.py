import numpy as np
from .base import ActivationFunc, T


def identity(x: T) -> T:
    """
    Identity function.
    :param x:
    :return:
    """
    return x


def identity_derivative(x: T) -> T:
    """
    Derivative of identity function.
    :param x:
    :return:
    """
    return np.ones(x.shape)


linear_activation = ActivationFunc(identity, identity_derivative, "linear")
