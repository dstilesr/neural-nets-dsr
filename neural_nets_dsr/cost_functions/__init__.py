from typing import Dict
from .base import CostFunction
from .mse_loss import mean_sq_error
from .logistic_loss import logistic_cost
from .multiclass_logistic_loss import mc_logistic_cost

COST_NAMES: Dict[str, CostFunction] = {
    logistic_cost.name: logistic_cost,
    mc_logistic_cost.name: mc_logistic_cost,
    mean_sq_error.name: mean_sq_error
}

AVAILABLE_COST_FUNCS = list(COST_NAMES.keys())
