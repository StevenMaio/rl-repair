from abc import ABC, abstractmethod


class FirstOrderMethod(ABC):

    def reset(self):
        ...

    def init(self, fprl: "FixPropRepairLearn"):
        ...

    @abstractmethod
    def step(self, fprl, gradient_estimate):
        ...
