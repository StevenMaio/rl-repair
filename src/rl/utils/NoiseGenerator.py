from typing import List, Iterator

import torch

from .TensorList import TensorList


class NoiseGenerator:
    _std_deviations: List[torch.Tensor]

    def __init__(self, tensors: Iterator[torch.Tensor]):
        self._std_deviations = [torch.ones_like(t) for t in tensors]

    def sample(self, generator=None, in_shared_mem=False, dropout_p=0.00):
        noises = []
        for std in self._std_deviations:
            epsilon = torch.normal(0,
                                   std=std,
                                   generator=generator)
            if torch.rand(1).item() <= dropout_p:
                epsilon.zero_()
            if in_shared_mem:
                epsilon.share_memory_()
            noises.append(epsilon)
        return TensorList(noises)
