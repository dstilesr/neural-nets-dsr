import numpy as np
from .base import CostFunction


def mse_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error cost for regressions.
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.mean(np.square(y_true - y_pred)).squeeze()


def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE cost.
    :param y_true:
    :param y_pred:
    :return:
    """
    return -2. * (y_true - y_pred)


mean_sq_error = CostFunction(mse_cost, mse_gradient)
