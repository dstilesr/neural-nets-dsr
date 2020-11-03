import numpy as np
from ..network import NeuralNet
from .base import UpdateStrategy
from typing import Union, List, Tuple
from ..cost_functions.base import CostFunction
from ..utils import ExpAvgAccumulator as ExpAvg
from .regularized_gradient_descent import GradientDescentL2


class RMSPropStrategy(UpdateStrategy):
    """
    Update parameters by Root Mean Square Propagation.
    """

    def __init__(
            self,
            learning_rate: float,
            l2_param: float = 0.01,
            beta: float = 0.99,
            normalize: bool = False,
            epsilon: float = 1e-6):

        assert 0.0 < beta < 0.1
        self.__learning_rate = learning_rate
        self.__l2_param = l2_param
        self.__beta = beta
        self.__accumulator: ExpAvg = None
        self.__normalize = normalize
        self.__epsilon = epsilon

    @property
    def lr(self) -> float:
        return self.__learning_rate

    @property
    def l2_param(self) -> float:
        return self.__l2_param

    @property
    def beta(self) -> float:
        return self.__beta

    @property
    def normalize(self) -> bool:
        return self.__normalize

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    def update_params(self, vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Update values by RMS prop.
        :param vals:
        :param grad:
        :return:
        """
        if self.__accumulator is None:
            self.__accumulator = ExpAvg.create(
                vals.shape,
                self.beta,
                self.normalize
            )

        self.__accumulator.update_value(np.square(grad))
        denom = np.sqrt(self.__accumulator.value + self.epsilon)
        full_grad = grad / denom + self.l2_param * vals
        return vals - self.lr * full_grad


class RMSProp(GradientDescentL2):
    """
    Root Mean Square Propagation Optimizer.
    """

    def __init__(
            self,
            cost_func: Union[str, CostFunction],
            epochs: int = 300,
            batch_size: int = -1,
            axis: int = 1,
            learning_rate: float = 0.1,
            l2_param: float = 0.005,
            beta: float = 0.99,
            epsilon: float = 1e-6,
            normalize_avg: bool = False,
            verbose: bool = False):
        """

        :param cost_func:
        :param epochs:
        :param batch_size:
        :param axis:
        :param learning_rate:
        :param l2_param:
        :param beta:
        :param normalize_avg:
        :param verbose:
        """

        self.__beta = beta
        self.__normalize = normalize_avg
        self.__epsilon = epsilon
        self.__rms_w: List[ExpAvg] = []
        self.__rms_b: List[ExpAvg] = []
        super().__init__(
            cost_func,
            epochs,
            batch_size,
            axis,
            learning_rate,
            l2_param,
            verbose
        )

    @property
    def beta(self) -> float:
        """
        Beta parameter for RMS Prop.
        :return:
        """
        return self.__beta

    def get_updates(
            self,
            w: np.ndarray,
            b: np.ndarray,
            dw: np.ndarray,
            db: np.ndarray,
            lyr_index: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RMS updates for weights and biases of a layer.
        :param w: Weights of the layer.
        :param b: Biases of the layer.
        :param dw: Gradient wrt weights.
        :param db: Gradient wrt biases.
        :param lyr_index: Index of the layer in the network.
        :return: The updated weights and biases.
        """
        self.__rms_w[lyr_index].update_value(np.square(dw))
        self.__rms_b[lyr_index].update_value(np.square(db))
        wreg = self.l2_param * w

        wnew = w - self.learning_rate * (
                dw / np.sqrt(self.__rms_w[lyr_index].value + self.__epsilon)
                + wreg
        )
        bnew = b - self.learning_rate * (
                db / np.sqrt(self.__rms_b[lyr_index].value + self.__epsilon)
        )
        return wnew, bnew

    def __call__(
            self,
            network: NeuralNet,
            x: np.ndarray,
            y: np.ndarray) -> NeuralNet:
        """
        Optimize the neural network with RMSProp.
        :param network:
        :param x: Train set features.
        :param y: Train set labels.
        :return:
        """
        for lyr in network.layers:
            self.__rms_w.append(ExpAvg.create(
                lyr.weights.shape,
                self.beta,
                self.__normalize
            ))
            self.__rms_b.append(ExpAvg.create(
                lyr.biases.shape,
                self.beta,
                self.__normalize
            ))

        return super().__call__(network, x, y)

