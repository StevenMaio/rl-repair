import torch
from typing import Tuple

from .ValueFixingStrategy import ValueFixingStrategy
from ...model import Model, Variable


class BadObjective(ValueFixingStrategy):

    def select_fixing_value(self,
                            model: Model,
                            var: Variable,
                            generator=None) -> Tuple[int, int]:
        if var.objective_coefficient > 0:
            return var.lb, var.ub
        elif var.objective_coefficient < 0:
            return var.ub, var.lb
        else:
            if torch.rand(1, generator=generator).item() <= 0.5:
                return var.ub, var.lb
            else:
                return var.lb, var.ub

    def init(self, model, generator=None):
        ...
