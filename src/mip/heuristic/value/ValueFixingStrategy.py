from abc import ABC, abstractmethod
from typing import Tuple


class ValueFixingStrategy(ABC):
    name: str = "AbstractValueFixingStrategy"

    @abstractmethod
    def select_fixing_value(self, model: "Model", var: "Variable", generator=None) -> Tuple[int, int]:
        """
        Determines the branching values to which the variable will be fixed.
        Returns a list [a, b] such that a and b are the branching values.
        :param generator:
        :param model:
        :param var:
        :return:
        """
        ...

    @abstractmethod
    def init(self, model, generator=None):
        """
        Perform preprocessing.
        :param generator:
        :param model:
        :return:
        """
        ...
