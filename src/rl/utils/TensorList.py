import torch
from typing import List, Iterator


class TensorList:
    _tensors: List[torch.Tensor]

    def __init__(self, tensors: Iterator[torch.Tensor]):
        self._tensors = list(tensors)

    def add_to_self(self, other: "TensorList"):
        for u, v in zip(self._tensors, other._tensors):
            u.add_(v)

    def add_to_iterator(self, other: Iterator[torch.Tensor]):
        for u, v in zip(self._tensors, other):
            v.add_(u)

    def scale(self, scale_factor: float):
        for t in self._tensors:
            t.multiply_(scale_factor)

    @staticmethod
    def zeros_like(tensor_sequence: Iterator[torch.Tensor]):
        zeros: List[torch.Tensor] = [torch.zeros_like(t) for t in tensor_sequence]
        return TensorList(zeros)
