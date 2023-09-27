from typing import Tuple

from .ValueFixingStrategy import ValueFixingStrategy
from ...model import Variable, Model


class SmallestViolationStrategy(ValueFixingStrategy):

    def select_fixing_value(self,
                            model: Model,
                            var: Variable,
                            generator=None) -> Tuple[int, int]:
        num_up_violations = 0
        num_down_violations = 0
        # compute the number of violated constraints when we fix the variable up
        column = var.column
        for cons_idx, coef in column:
            cons = model.get_constraint(cons_idx)
            # compute potential up lock
            if coef > 0:
                max_activity = cons.max_activity
                min_activity = cons.min_activity + (var.ub - var.lb) * coef
            else:
                max_activity = cons.max_activity + (var.ub - var.lb) * coef
                min_activity = cons.min_activity
            if cons.is_violated(min_activity, max_activity):
                num_up_violations += 1
            # compute potential down lock
            if coef > 0:
                max_activity = cons.max_activity + (var.lb - var.ub) * coef
                min_activity = cons.min_activity
            else:
                max_activity = cons.max_activity
                min_activity = cons.min_activity + (var.lb - var.ub) * coef
            if cons.is_violated(min_activity, max_activity):
                num_down_violations += 1
        if num_down_violations < num_down_violations:
            return var.lb, var.ub
        else:
            return var.ub, var.lb

    def init(self, model, generator=None):
        ...