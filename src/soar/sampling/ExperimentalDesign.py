from abc import ABC, abstractmethod

import torch


class ExperimentalDesign(ABC):

    @abstractmethod
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def support(self) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def num_dimensions(self) -> int:
        ...
