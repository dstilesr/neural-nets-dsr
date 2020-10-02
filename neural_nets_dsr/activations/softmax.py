import numpy as np
from .base import ActivationFunc, T


def softmax_func(x: T) -> T:
    """
    Softmax function for multiclass classification.
    :param x:
    :return:
    """
    d = np.max(x, axis=0, keepdims=True)
    x_exp = np.exp(x - d)
    return x_exp / np.sum(x_exp, axis=0, keepdims=True)


def softmax_gradient(x: T) -> T:
    """
    Gradient of softmax function.
    :param x:
    :return:
    """
    sm = softmax_func(x)
    return sm - np.square(sm)


softmax_activation = ActivationFunc(softmax_func, softmax_gradient)
