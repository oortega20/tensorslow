from abc import ABC
from abc import abstractmethod


class Dataset(ABC):
    """Abstract class for dataset objects"""
    def __init__(self):
        self._data = self.download()
        self._x_train = self._data['x_train']
        self._y_train = self._data['y_train']
        self._x_test = self._data['x_test']
        self._y_test = self._data['y_test']


    @abstractmethod
    def download(self):
        """download data to use within object"""
        pass

    def get_train_data(self):
        """get dataset object training data"""
        return self._x_train, self._y_train

    def get_test_data(self):
        """get dataset object testing data"""
        return self._x_test, self._y_test


