from abc import ABC, abstractmethod


class GradientEstimator(ABC):

    @abstractmethod
    def estimate_gradient(self, instances, fprl):
        ...
