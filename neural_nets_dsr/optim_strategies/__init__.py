from typing import Dict, Type
from .base import UpdateStrategy
from .rmsprop import RMSPropStrategy
from .adam import AdamStrategy
from .gradient_descent import GradientDescentStrategy
from .gradient_descent_momentum import GDMomentumStrategy
from .regularized_gradient_descent import GradientDescentL2Strategy


STRATEGY_NAMES: Dict[str, Type[UpdateStrategy]] = {
    GradientDescentStrategy.NAME: GradientDescentStrategy,
    GradientDescentL2Strategy.NAME: GradientDescentL2Strategy,
    GDMomentumStrategy.NAME: GDMomentumStrategy,
    RMSPropStrategy.NAME: RMSPropStrategy,
    AdamStrategy.NAME: AdamStrategy
}

