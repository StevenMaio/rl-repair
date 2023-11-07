from typing import List

from .FixingOrderStrategy import FixingOrderStrategy
from ...model import Model, Variable, VarType


class VariableLocksOrder(FixingOrderStrategy):
    """
    This strategy sorts the variables in non-increasing order based on the
    number of variables locks.
    """
    name: str = "VariableLocksOrder"
    _indices: List[int]

    def __init__(self):
        self._indices = []

    def init(self, model, generator=None):
        self._indices.clear()

        variables = model.variables.copy()
        variables.sort(key=lambda x: x.num_up_locks + x.num_down_locks,
                       reverse=True)
        v: Variable
        for v in filter(lambda x: x.type != VarType.CONTINUOUS,
                        variables):
            self._indices.append(v.id)

    def select_variable(self, model: Model, generator=None) -> Variable:
        idx: int
        for idx in self._indices:
            var: Variable = model.get_var(idx)
            if var.lb != var.ub:
                return var
