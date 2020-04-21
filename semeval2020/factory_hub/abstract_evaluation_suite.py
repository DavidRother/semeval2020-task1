from abc import ABC, abstractmethod


class AbstractEvaluationSuite(ABC):

    @abstractmethod
    def evaluate(self, predictions, truth):
        raise NotImplementedError()
