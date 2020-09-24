from .base import CostFunction
from .mse_loss import mean_sq_error
from .logistic_loss import logistic_cost

COST_NAMES = {
    "logistic": logistic_cost,
    "mse": mean_sq_error
}
