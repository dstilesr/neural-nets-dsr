import numpy as np
from .base import ActivationFunc, T


def sigmoid(x: T) -> T:
    """
    Sigmoid function x -> 1 / (1 + exp(-x))
    :param x:
    :return:
    """
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x: T) -> T:
    """
    Derivative of sigmoid function x -> sigmoid(x) * (1 - sigmoid(x))
    :param x:
    :return:
    """
    sig = sigmoid(x)
    return sig - np.square(sig)


sigmoid_activation = ActivationFunc(sigmoid, sigmoid_derivative)
