from .base import ActivationFunc
from .relu import relu_activation
from .tanh import tanh_activation
from .sigmoid import sigmoid_activation
from .identity import linear_activation

ACTIVATIONS_NAMES = {
    "relu": relu_activation,
    "tanh": tanh_activation,
    "sigmoid": sigmoid_activation,
    "linear": linear_activation
}
