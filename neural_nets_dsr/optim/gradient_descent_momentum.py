import numpy as np
from typing import Union, List
from ..network import NeuralNet
from ..cost_functions import CostFunction
from ..utils import ExpAvgAccumulator as ExpAvg
from .regularized_gradient_descent import GradientDescentL2


class GradientDescentWithMomentum(GradientDescentL2):
    """
    Mini batch gradient descent with momentum.
    """

    def __init__(
            self, cost_func: Union[str, CostFunction],
            epochs: int = 600,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            batch_size: int = 512,
            beta: float = 0.9,
            axis: int = 1,
            verbose: bool = False):
        """

        :param cost_func: Cost function to optimize.
        :param epochs: Number of full train set passes to perform.
        :param learning_rate:
        :param l2_param: Parameter for L2 regularization.
        :param batch_size: Minibatch size.
        :param beta: Meta parameter for momentum term.
        :param axis:
        :param verbose: Print cost every 100 epochs.
        """
        assert 0. < beta < 1., "Invalid beta parameter! Must satisfy 0 < beta < 1."
        super().__init__(
            cost_func,
            epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2_param=l2_param,
            axis=axis,
            verbose=verbose
        )
        self._batch_size = batch_size
        self._beta = beta
        self._momentum_b: List[ExpAvg] = []
        self._momentum_w: List[ExpAvg] = []

    @property
    def beta(self) -> float:
        return self._beta

    def gradient_descent_iteration(
            self,
            x: np.ndarray,
            y: np.ndarray) -> float:
        if self._network is None:
            raise NotImplementedError("No network selected!")

        y_pred = self._network.compute_predictions(x, True)
        cost = self.cost_func(y, y_pred)
        da = self.cost_func.gradient(y, y_pred)

        for i in reversed(range(len(self._network.layers))):
            lyr = self._network.layers[i]
            dw, db, da = lyr.back_prop(da)

            self._momentum_b[i].update_value(db)
            self._momentum_w[i].update_value(dw)

            reg_w = self.l2_param * lyr.weights
            lyr.set_weights(
                w=lyr.weights
                    - self.learning_rate * self._momentum_w[i].value - reg_w,
                b=lyr.biases - self.learning_rate * self._momentum_b[i].value
            )
        return cost

    def __call__(
            self,
            network: NeuralNet,
            x: np.ndarray,
            y: np.ndarray) -> NeuralNet:
        """

        :param network:
        :param x:
        :param y:
        :return:
        """
        for lyr in network.layers:
            self._momentum_w.append(
                ExpAvg.create(lyr.weights.shape, self.beta)
            )
            self._momentum_b.append(
                ExpAvg.create(lyr.biases.shape, self.beta)
            )

        return super().__call__(network, x, y)
