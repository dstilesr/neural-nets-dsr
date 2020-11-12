import numpy as np
from abc import ABC, abstractmethod
from typing import Collection, Tuple


class BatchStrategy(ABC):
    """
    Batch strategies extract a minibatch from a dataset for training.
    """

    @abstractmethod
    def extract(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            indices: Collection[int]) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    @abstractmethod
    def examples_axis(self) -> int:
        pass


class ConvStrategy(BatchStrategy):
    """
    Extract batch from 4D arrays used in 2D convnets.
    """

    @property
    def examples_axis(self) -> int:
        return 0

    def extract(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            indices: Collection[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract next batch for a convnet.
        :param xs: Features. Shape: (examples, sizex, sizey, channels).
        :param ys: Labels. Shape: (output_dim, examples).
        :param indices: Indices to extract.
        :return: Minibatch features and labels.
        """

        return xs[indices, :, :, :], ys[:, indices]


class DenseStrategy(BatchStrategy):
    """
    Extract batch from 2D arrays used in simple dense networks.
    """

    @property
    def examples_axis(self) -> int:
        return 1

    def extract(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            indices: Collection[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get next batch.
        :param xs: Features. Shape: (num_features, examples).
        :param ys: Labels. Shape: (output_dim, examples).
        :param indices: Indices of examples to extract.
        :return: Minibatch features and labels.
        """

        return xs[:, indices], ys[:, indices]


class RecurrentStrategy(BatchStrategy):
    """
    Extract batch from 3D arrays used for RNNs.
    """

    @property
    def examples_axis(self) -> int:
        return 0

    def extract(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            indices: Collection[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get next batch.
        :param xs: Features. Shape: (examples, sequence_length, num_features).
        :param ys: Labels. Shape: (output_dim, examples).
        :param indices: Indices of examples to extract.
        :return: Minibatch features and labels.
        """

        return xs[indices, :, :], ys[:, indices]
