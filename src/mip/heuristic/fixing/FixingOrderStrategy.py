"""
Classes representing the order in which variables are fixed. There is hopefully
a cool, sexy way to do this efficiently. But I'm not smart enough to see it.

Author: Steven Maio
"""

from abc import ABC, abstractmethod

from src.mip.model import Variable, Model


class FixingOrderStrategy(ABC):
    name: str

    @abstractmethod
    def init(self, model, generator=None):
        """
        Perform preprocessing.
        :param generator:
        :param model:
        :return:
        """
        ...

    @abstractmethod
    def select_variable(self, model: Model, generator=None) -> "Variable":
        """
        Computes the next variable to be fixed. Should return None if all the
        integer variables have been fixed.
        :param generator:
        :param model:
        :return: the next variable to be fixed.
        """
        ...
