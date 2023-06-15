from abc import ABC, abstractmethod


class LearningAlgorithm(ABC):

    @abstractmethod
    def train(self, fprl, instances):
        ...
