from typing import List

from .FixingOrderStrategy import FixingOrderStrategy
from ...model import Model, VarType, Variable


class TypeFixingOrder(FixingOrderStrategy):
    name: str = 'TypeFixingOrder'

    _indices: List[int]

    def __init__(self):
        self._indices = []

    def init(self, model, generator=None):
        self._indices.clear()
        binary_indices = []
        integer_indices = []
        for var in model.variables:
            if var.type == VarType.BINARY:
                binary_indices.append(var.id)
            if var.type == VarType.INTEGER:
                integer_indices.append(var.id)
        self._indices.extend(binary_indices)
        self._indices.extend(integer_indices)

    def select_variable(self, model: Model, generator=None) -> Variable:
        for idx in self._indices:
            var = model.get_var(idx)
            if var.lb != var.ub:
                return var
