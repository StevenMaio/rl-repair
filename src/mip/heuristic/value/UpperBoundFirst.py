from typing import Tuple

from src.mip.heuristic.value.ValueFixingStrategy import ValueFixingStrategy


class UpperBoundFirst(ValueFixingStrategy):
    name: str = "UpperBoundFirst"

    def select_fixing_value(self, model: "Model", var: "Variable", generator=None) -> Tuple[int, int]:
        return var.ub, var.lb

    def init(self, model, generator=None):
        ...
