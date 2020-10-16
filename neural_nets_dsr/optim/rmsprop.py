import numpy as np
from typing import Union, List
from ..network import NeuralNet
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

    def gradient_descent_iteration(
            self,
            x: np.ndarray,
            y: np.ndarray) -> float:
        """
        Performs iteration of gradient descent with momentum.
        :param x:
        :param y:
        :return:
        """
        if self._network is None:
            raise NotImplementedError("No network selected!")

        y_pred = self._network.compute_predictions(x, True)
        cost = self.cost_func(y, y_pred)
        da = self.cost_func.gradient(y, y_pred)

        for i in reversed(range(len(self._network.layers))):
            lyr = self._network.layers[i]
            dw, db, da = lyr.back_prop(da)

            self.__rms_b[i].update_value(np.square(db))
            self.__rms_w[i].update_value(np.square(dw))
            denom_b = db / np.sqrt(self.__rms_b[i].value + self.__epsilon)
            denom_w = dw / np.sqrt(self.__rms_w[i].value + self.__epsilon)

            reg_w = self.learning_rate * self.l2_param * lyr.weights
            lyr.set_weights(
                w=lyr.weights - self.learning_rate * dw / denom_w - reg_w,
                b=lyr.biases - self.learning_rate * db / denom_b
            )
        return cost

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

