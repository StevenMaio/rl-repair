from typing import Tuple

import torch

from .ValueFixingStrategy import ValueFixingStrategy
from ...model import Model, Variable


class LpFixingStrategy(ValueFixingStrategy):

    def select_fixing_value(self, model: Model, var: Variable, generator=None) -> Tuple[int, int]:
        fractional_val = var.relaxation_value % 1
        if torch.rand(1, generator=generator).item() <= fractional_val:
            return var.ub, var.lb
        else:
            return var.lb, var.ub

    def init(self, model, generator=None):
        model.solve_relaxation()
