from abc import ABC, abstractmethod


class Objective(ABC):

    @abstractmethod
    def evaluate(self):
        ...
