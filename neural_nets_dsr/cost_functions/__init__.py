from .base import CostFunction
from .mse_loss import mean_sq_error
from .logistic_loss import logistic_cost
from .multiclass_logistic_loss import mc_logistic_cost

COST_NAMES = {
    "logistic": logistic_cost,
    "multiclass_logistic": mc_logistic_cost,
    "mse": mean_sq_error
}
