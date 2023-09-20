from abc import ABC, abstractmethod
from typing import Union

import torch


class TerminationMechanism(ABC):

    @abstractmethod
    def init(self, start_point: torch.Tensor):
        ...

    @abstractmethod
    def update(self,
               next_point: torch.Tensor,
               observed_value: Union[torch.Tensor, float],
               computational_cost: int):
        ...

    @abstractmethod
    def should_stop(self) -> bool:
        ...
