from typing import Tuple

from .ValueFixingStrategy import ValueFixingStrategy
from ...model import Variable, Model


class LeastLocksStrategy(ValueFixingStrategy):

    def select_fixing_value(self,
                            model: Model,
                            var: Variable,
                            generator=None) -> Tuple[int, int]:
        if var.num_up_locks > var.num_down_locks:
            return var.lb, var.ub
        else:
            return var.ub, var.lb

    def init(self, model, generator=None):
        ...
