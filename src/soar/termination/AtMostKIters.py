"""
A termination mechanism that ends a local search after a maximum number of
iterations is performed.
"""
from typing import Union
import torch

from .TerminationMechanism import TerminationMechanism


class AtMostKIters(TerminationMechanism):
    _max_iters: int
    _num_iters: int

    def __init__(self, max_iters):
        self._max_iters = max_iters
        self._num_iters = 0

    def init(self, start_point: torch.Tensor):
        self._num_iters = 0

    def update(self,
               next_point: torch.Tensor,
               observed_value: Union[torch.Tensor, float],
               computational_cost: int):
        self._num_iters += 1

    def should_stop(self) -> bool:
        return self._num_iters == self._max_iters
