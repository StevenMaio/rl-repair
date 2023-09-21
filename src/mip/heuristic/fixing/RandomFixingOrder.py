from typing import List

import torch

from src.mip.heuristic.fixing import FixingOrderStrategy
from src.mip.model import Model, Variable, VarType


class RandomFixingOrder(FixingOrderStrategy):
    """
    This shit is kinda hard to do... I don't really know a good way to do this.
    """

    name: str = 'RandomFixingOrder'

    _indices: List[int]

    def __init__(self):
        self._indices = []

    def select_variable(self, model: Model, generator=None) -> Variable:
        for idx in self._indices:
            var = model.get_var(idx)
            if var.lb != var.ub:
                return var

    def init(self, model, generator=None):
        self._indices.clear()
        binary_indices = []
        integer_indices = []
        for var in model.variables:
            if var.type == VarType.BINARY:
                binary_indices.append(var.id)
            if var.type == VarType.INTEGER:
                integer_indices.append(var.id)
        permutation = torch.randperm(len(binary_indices))
        permutated_binary_indices = [binary_indices[idx] for idx in permutation]
        self._indices.extend(permutated_binary_indices)

        permutation = torch.randperm(len(integer_indices))
        permutated_integer_indices = [integer_indices[idx] for idx in permutation]
        self._indices.extend(permutated_integer_indices)
