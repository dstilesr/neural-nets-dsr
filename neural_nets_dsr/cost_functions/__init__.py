from .base import CostFunction
from .logistic_loss import logistic_cost

COST_NAMES = {
    "logistic": logistic_cost
}
