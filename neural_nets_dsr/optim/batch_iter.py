import numpy as np
from typing import Iterable, Tuple
from abc import ABC, abstractmethod


class BaseBatchIter(ABC, Iterable[Tuple[np.ndarray, np.ndarray]]):
    """
    Abstract batch iterator
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, axis: int = 1):
        """

        :param x:
        :param y:
        :param axis: Axis corresponding to different examples.
        """
        self._x = x
        self._y = y
        self._axis = axis
        self._ndim = x.ndim

    @property
    def axis(self) -> int:
        return self._axis

    @abstractmethod
    def reset_iterator(self):
        pass

    @abstractmethod
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def __iter__(self):
        return self


class FullBatchIterator(BaseBatchIter):
    """
    Batch iterator. Gives the full batch at each step.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            axis: int = 1,
            epochs: int = 150):
        """
        Initialize the batch iterator.
        :param x: Train set features.
        :param y: Train set labels.
        :param axis: Axis of examples.
        :param epochs: Number of iterations.
        """
        super().__init__(x, y, axis)
        self._total_epochs = epochs
        self._current_epoch = 0

    @property
    def epochs(self) -> int:
        """
        Number of epochs.
        :return:
        """
        return self._total_epochs

    def reset_iterator(self):
        """
        Restart the iterator.
        :return:
        """
        self._current_epoch = 0

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._current_epoch >= self._total_epochs:
            raise StopIteration

        self._current_epoch += 1
        return self._x, self._y


class MiniBatchIterator(BaseBatchIter):
    """
    Iterate on minibatches of the train set.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            axis: int = 1,
            batch_size: int = 64,
            epochs: int = 150,
            shuffle: bool = True,
            seed: int = 951):
        """

        :param x:
        :param y:
        :param axis: Axis for splitting minibatches.
        :param batch_size: Size of minibatches.
        :param epochs:
        :param shuffle: Shuffle indices.
        :param seed: Seed (for shuffle).
        """
        self.check_axis(axis)
        super().__init__(x, y, axis)
        self._total_epochs = epochs
        self._current_epoch = 0
        self._current_index = 0
        self._batch_size = batch_size
        self.__indices = np.arange(0, x.shape[axis])

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.__indices)

    @staticmethod
    def check_axis(axis: int):
        """
        Checks whether the axis is within supported values (currently 0 - rows
        or 1 - columns).
        :param axis:
        :return:
        """
        if axis != 0 and axis != 1:
            raise ValueError("Invalid axis! Supported values: 0, 1.")

    @property
    def epochs(self) -> int:
        return self._total_epochs

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def reset_iterator(self):
        self._current_index = 0
        self._current_epoch = 0

    def next_indices(self) -> np.ndarray:
        """
        Gives the indices for the next minibatch.
        :return:
        """
        if self._current_epoch >= self._total_epochs:
            raise StopIteration

        stop_ind = self._current_index + self.batch_size
        if stop_ind >= self.__indices.shape[0]:
            self._current_epoch += 1
            inds = self.__indices[self._current_index:self.__indices.shape[0]]
            self._current_index = 0
        else:
            inds = self.__indices[self._current_index:stop_ind]
            self._current_index = stop_ind
        return inds

    def __next__(self):
        inds = self.next_indices()
        if self._ndim == 2:
            if self.axis == 0:
                return self._x[inds, :], self._y[inds, :]
            else:
                return self._x[:, inds], self._y[:, inds]
        elif self._ndim == 4:
            assert self.axis == 0
            return self._x[inds, :, :, :], self._y[:, inds]
