from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError()

    @abstractmethod
    def fit_predict(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def fit_predict_labeling(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict_with_extra_return(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict_labeling(self, data, **kwargs):
        raise NotImplementedError()
