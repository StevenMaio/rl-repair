from typing import List

from src.mip.heuristic.fixing import FixingOrderStrategy
from src.mip.model import Model, Variable, VarType


class LeftRightOrder(FixingOrderStrategy):
    """
    In the original FPR paper, LR and type are different. However, we sort
    variables by type when we instantiate a Model instance using Gurobi. S0
    in that case, they're equivalent here.
    """

    name: str = "LeftRightOrder"

    _indices: List[int]

    def __init__(self):
        self._indices = []

    def select_variable(self, model: Model, generator=None) -> "Variable":
        for idx in self._indices:
            var = model.get_var(idx)
            if var.lb != var.ub:
                return var

    def init(self, model: Model, generator=None):
        self._indices.clear()
        for var in model.variables:
            if var.type == VarType.BINARY or var.type == VarType.INTEGER:
                self._indices.append(var.id)
