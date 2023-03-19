from abc import ABC
from abc import abstractmethod


class Dataset(ABC):
    def __init__(self):
        self._data = self.download()
        self._x_train = self._data['x_train']
        self._y_train = self._data['y_train']
        self._x_test = self._data['x_test']
        self._y_test = self._data['y_test']


    @abstractmethod
    def download(self):
        pass

    def get_train_data(self):
        return self._x_train, self._y_train

    def get_test_data(self):
        return self._x_test, self._y_test


