import numpy as np
from .base import CostFunction


def logistic_cost_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Logistic cost function for binary classifiers.
    :param y_true:
    :param y_pred:
    :return:
    """
    losses = -(y_true * np.log(y_pred) + (1. - y_true) * np.log(1 - y_pred))
    return np.squeeze(np.mean(losses))


def logistic_loss_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return -y_true / y_pred + (1. - y_true) / (1. - y_pred)


logistic_cost = CostFunction(logistic_cost_func, logistic_loss_grad)
