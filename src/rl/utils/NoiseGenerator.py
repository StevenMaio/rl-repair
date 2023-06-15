from typing import List, Iterator

import torch

from .TensorList import TensorList
from torch.distributions import Normal


class NoiseGenerator:
    _distributions: List[Normal]

    def __init__(self, tensors: Iterator[torch.Tensor]):
        self._distributions = [Normal(torch.zeros_like(t), torch.ones_like(t)) for t in tensors]

    def sample(self):
        noises = []
        for distribution in self._distributions:
            noises.append(distribution.sample())
        return TensorList(noises)
