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
        for var in model.variables:
            if var.type == VarType.BINARY or var.type == VarType.INTEGER:
                self._indices.append(var.id)
        # permutate the matrix
        permutation = torch.randperm(len(self._indices))
        permutated_indices = [self._indices[idx] for idx in permutation]
        self._indices = permutated_indices
