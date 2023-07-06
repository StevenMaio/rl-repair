from abc import ABC, abstractmethod


class FirstOrderMethod(ABC):

    def reset(self):
        ...

    @abstractmethod
    def step(self, fprl, gradient_estimate):
        ...
