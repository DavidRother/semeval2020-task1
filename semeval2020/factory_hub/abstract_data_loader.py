from abc import ABC, abstractmethod


class AbstractDataLoader(ABC):

    @abstractmethod
    def load(self):
        raise NotImplementedError()
