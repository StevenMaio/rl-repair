from typing import List, Iterator

import torch

from .TensorList import TensorList


class NoiseGenerator:
    _std_deviations: List[torch.Tensor]

    def __init__(self, tensors: Iterator[torch.Tensor]):
        self._std_deviations = [torch.ones_like(t) for t in tensors]

    def sample(self, generator=None, in_shared_mem=False):
        noises = []
        for std in self._std_deviations:
            epsilon = torch.normal(0,
                                   std=std,
                                   generator=generator)
            if in_shared_mem:
                epsilon.share_memory_()
            noises.append(epsilon)
        return TensorList(noises)
