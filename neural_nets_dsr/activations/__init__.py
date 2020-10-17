from typing import Dict
from .base import ActivationFunc
from .relu import relu_activation
from .tanh import tanh_activation
from .sigmoid import sigmoid_activation
from .identity import linear_activation
from .softmax import softmax_activation

ACTIVATIONS_NAMES: Dict[str, ActivationFunc] = {
    relu_activation.name: relu_activation,
    tanh_activation.name: tanh_activation,
    sigmoid_activation.name: sigmoid_activation,
    linear_activation.name: linear_activation,
    softmax_activation.name: softmax_activation
}

AVAILABLE_ACTIVATIONS = list(ACTIVATIONS_NAMES.keys())
