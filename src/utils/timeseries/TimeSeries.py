from abc import ABC, abstractmethod


class TimeSeries(ABC):

    @abstractmethod
    def reset(self, hard_reset=False):
        ...

    @property
    def level(self):
        raise NotImplementedError()
