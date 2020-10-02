import numpy as np
from typing import Union
from ..network import NeuralNet
from ..cost_functions import CostFunction
from .minibatch_gradient_descent import MiniBatchGDL2


class AdamOptimizer(MiniBatchGDL2):
    """
    Mini batch gradient descent with momentum.
    """

    def __init__(
            self, cost_func: Union[str, CostFunction],
            epochs: int = 600,
            learning_rate: float = 0.1,
            l2_param: float = 0.025,
            batch_size: int = 512,
            beta_momentum: float = 0.9,
            beta_rms: float = 0.99,
            epsilon: float = 1e-7,
            verbose: bool = False):
        """

        :param cost_func:
        :param epochs:
        :param learning_rate:
        :param l2_param:
        :param batch_size:
        :param beta_momentum: Beta for momentum parameter.
        :param beta_rms: Beta for RMS.
        :param epsilon:
        :param verbose:
        """
        assert 0. < beta_momentum < 1., "Invalid beta parameter! Must satisfy 0 < beta < 1."
        assert 0. < beta_rms < 1., "Invalid beta parameter! Must satisfy 0 < beta < 1."
        super().__init__(
            cost_func,
            epochs,
            learning_rate,
            l2_param,
            batch_size,
            verbose=verbose
        )
        self._batch_size = batch_size
        self._beta_mom = beta_momentum
        self._beta_rms = beta_rms
        self._momentum_b = []
        self._rms_b = []
        self._momentum_w = []
        self._rms_w = []
        self._epsilon = epsilon
        self.__iter = 0

    @property
    def beta_momentum(self) -> float:
        return self._beta_mom

    @property
    def beta_rms(self) -> float:
        return self._beta_rms

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def update_momentum_rms(
            self,
            dw: np.ndarray,
            db: np.ndarray,
            lyr_num: int):
        """
        Updates the momentum and RMS prop parameters with the current
        gradients.
        :param dw:
        :param db:
        :param lyr_num:
        :return:
        """
        self._momentum_w[lyr_num] = (
            self.beta_momentum * self._momentum_w[lyr_num]
            + (1 - self.beta_momentum) * dw
        )

        self._momentum_b[lyr_num] = (
                self.beta_momentum * self._momentum_b[lyr_num]
                + (1 - self.beta_momentum) * db
        )
        # RMS Param updates
        self._rms_w[lyr_num] = (
            self.beta_rms * self._rms_w[lyr_num]
            + (1 - self.beta_rms) * np.square(dw)
        )

        self._rms_b[lyr_num] = (
                self.beta_rms * self._rms_b[lyr_num]
                + (1 - self.beta_rms) * np.square(db)
        )

        # Normalization for numerical stability
        self._momentum_w[lyr_num] /= (1 - self.beta_momentum ** self.__iter)
        self._momentum_b[lyr_num] /= (1 - self.beta_momentum ** self.__iter)
        self._rms_w[lyr_num] /= (1 - self.beta_rms ** self.__iter)
        self._rms_b[lyr_num] /= (1 - self.beta_rms ** self.__iter)

    def gradient_descent_iteration(
            self,
            x: np.ndarray,
            y: np.ndarray) -> float:
        """

        :param x:
        :param y:
        :return:
        """
        self.__iter += 1
        if self._network is None:
            raise NotImplementedError("No network selected!")

        y_pred = self._network.compute_predictions(x, True)
        cost = self.cost_func(y, y_pred)
        da = self.cost_func.gradient(y, y_pred)

        for i in reversed(range(len(self._network.layers))):
            lyr = self._network.layers[i]
            dw, db, da = lyr.back_prop(da)
            self.update_momentum_rms(dw, db, i)

            reg_w = self.l2_param * lyr.weights
            lyr.set_weights(
                w=(
                    lyr.weights
                    - reg_w
                    - self.learning_rate * self._momentum_w[i] / (
                        self.epsilon + np.sqrt(self._rms_w[i])
                    )
                ),
                b=(
                    lyr.biases
                    - self.learning_rate * self._momentum_b[i] / (
                        self.epsilon + np.sqrt(self._rms_b[i])
                    )
                )
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
            self._momentum_w.append(np.zeros(lyr.weights.shape))
            self._momentum_b.append(np.zeros(lyr.biases.shape))
            self._rms_w.append(np.zeros(lyr.weights.shape))
            self._rms_b.append(np.zeros(lyr.biases.shape))

        return super().__call__(network, x, y)
