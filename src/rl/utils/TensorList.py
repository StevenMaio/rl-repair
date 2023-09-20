"""
A class that aggregates a list of tensors so as to treat it as a single vector.

References:
"""
import torch
from typing import List, Iterator


class TensorList:
    _tensors: List[torch.Tensor]
    _size: int

    def __init__(self, tensors: Iterator[torch.Tensor], clone=False):
        """
        Creates a new TensorList object from an iterator of Tensor objects. If
        clone is set to true, then the data is cloned. Thus, operations will
        not affect the original list of Tensor instances.
        :param tensors:
        :param clone:
        """
        self._size = 0
        if not clone:
            self._tensors = list(tensors)
            for t in self._tensors:
                self._size += t.shape.numel()
        else:
            self._tensors = []
            for t in tensors:
                self._size += t.shape.numel()
                self._tensors.append(t.clone())

    def add_to_self(self, other: "TensorList"):
        for u, v in zip(self._tensors, other._tensors):
            if u is not None and v is not None:
                u.add_(v)

    def add_to_iterator(self, other: Iterator[torch.Tensor]):
        for u, v in zip(self._tensors, other):
            if u is not None and v is not None:
                v.add_(u)

    def add_from_iterator(self, other: Iterator[torch.Tensor]):
        for u, v in zip(self._tensors, other):
            if u is not None and v is not None:
                u.add_(v)

    def scale(self, scale_factor: float):
        for t in self._tensors:
            t.multiply_(scale_factor)

    @staticmethod
    def zeros_like(tensor_sequence: Iterator[torch.Tensor]):
        zeros: List[torch.Tensor] = [torch.zeros_like(t, requires_grad=False) for t in tensor_sequence]
        return TensorList(zeros)

    @staticmethod
    def ones_like(tensor_sequence: Iterator[torch.Tensor]):
        ones: List[torch.Tensor] = [torch.ones_like(t, requires_grad=False) for t in tensor_sequence]
        return TensorList(ones)

    def zero_out(self):
        for t in self._tensors:
            t.zero_()

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._tensors)

    def clone(self) -> "TensorList":
        copied_data = [torch.clone(t) for t in self]
        return TensorList(copied_data)

    def copy_(self, src: "TensorList"):
        """
        An in-place copy operation. The data of the tensors inside data will be copied
        into the tensors of self._data. This is a direct analog of torch.clone_ [1].
        
        References:
            [1] https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_
        """
        u: torch.Tensor
        v: torch.Tensor
        for u, v in zip(self, src):
            u.copy_(v)

    def copy_into_iter(self, dest: Iterator[torch.Tensor]):
        u: torch.Tensor
        v: torch.Tensor
        for u, v in zip(self, dest):
            v.copy_(u)

    def copy_from_1d_tensor(self, src: torch.Tensor):
        with torch.no_grad():
            current_idx = 0
            for p in self:
                p_size = len(p.flatten())
                p.flatten().copy_(src[current_idx: current_idx + p_size])
                current_idx += p_size

    def flatten(self) -> torch.Tensor:
        """
        An analog of the method implemented in PyTorch. Takes a TensorList instance
        and copies the data into a 1d Tensor. This flattened Tensor is returned
        :return:
        """
        size = 0
        for p in self:
            size += len(p.flatten())
        t = torch.Tensor(size)
        current_idx = 0
        for p in self:
            p_size = len(p.flatten())
            t[current_idx: current_idx + p_size] = p.flatten()
            current_idx += p_size
        return t

    @property
    def size(self):
        return self._size
