import numpy as np
from .base import ActivationFunc, T


def softmax_func(x: T) -> T:
    """
    Softmax function for multiclass classification.
    :param x:
    :return:
    """
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=0)


def softmax_gradient(x: T) -> T:
    """
    Gradient of softmax function.
    :param x:
    :return:
    """
    sm = softmax_func(x)
    return sm - sm ** 2


softmax_activation = ActivationFunc(softmax_func, softmax_gradient)
