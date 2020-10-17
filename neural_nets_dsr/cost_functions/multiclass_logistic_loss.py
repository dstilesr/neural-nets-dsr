import numpy as np
from .base import CostFunction


def mc_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Logistic loss for multiclass classifiers.
    :param y_true:
    :param y_pred:
    :return:
    """
    return - np.mean(np.sum(np.log(y_pred) * y_true, axis=0)).squeeze()


def mc_logloss_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of multiclass logistic loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    return -y_true / y_pred


mc_logistic_cost = CostFunction(mc_logloss, mc_logloss_grad, "multiclass_logistic")
