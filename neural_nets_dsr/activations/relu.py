from .base import ActivationFunc, T


def relu(x: T) -> T:
    """
    Rectified Linear Unit function x -> max(x, 0)
    :param x:
    :return:
    """
    return x * (x >= 0)


def relu_derivative(x: T) -> T:
    """
    Derivative of ReLU function.
    :param x:
    :return:
    """
    return x >= 0


relu_activation = ActivationFunc(relu, relu_derivative, "relu")
