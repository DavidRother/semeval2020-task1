from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError()
