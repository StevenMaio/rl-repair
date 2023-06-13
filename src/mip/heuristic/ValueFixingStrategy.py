from abc import ABC, abstractmethod
from typing import List, Tuple

import random


class ValueFixingStrategy(ABC):
    name: str = "AbstractValueFixingStrategy"

    @abstractmethod
    def select_fixing_value(self, model: "Model", var: "Variable") -> Tuple[int, int]:
        """
        Determines the branching values to which the variable will be fixed.
        Returns a list [a, b] such that a and b are the branching values.
        :param model:
        :param var:
        :return:
        """
        ...


class RandomValueFixing(ValueFixingStrategy):
    name: str = "RandomValueFixing"

    def select_fixing_value(self, model: "Model", var: "Variable") -> Tuple[int, int]:
        local_domain: "Domain" = var.local_domain
        lower_bound: int = int(local_domain.lower_bound)
        upper_bound: int = int(local_domain.upper_bound)
        if random.random() <= 0.5:
            left_value, right_value = lower_bound, upper_bound
        else:
            left_value, right_value = upper_bound, lower_bound
        return left_value, right_value


class UpperBoundFirst(ValueFixingStrategy):
    name: str = "UpperBoundFirst"

    def select_fixing_value(self, model: "Model", var: "Variable") -> Tuple[int, int]:
        return var.ub, var.lb

