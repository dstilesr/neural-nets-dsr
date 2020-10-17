import numpy as np
from ..network import NeuralNet
from typing import Union, List, Tuple
from ..cost_functions.base import CostFunction
from ..utils import ExpAvgAccumulator as ExpAvg
from .regularized_gradient_descent import GradientDescentL2


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

