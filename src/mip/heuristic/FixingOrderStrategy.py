"""
Classes representing the order in which variables are fixed. There is hopefully
a cool, sexy way to do this efficiently. But I'm not smart enough to see it.

TODO:
    - [ ] think of a more efficient way to select a variable

Author: Steven Maio
"""
import torch

from typing import List
from abc import ABC, abstractmethod

from src.mip.model import VarType, Variable


class FixingOrderStrategy(ABC):
    name: str

    @abstractmethod
    def select_variable(self, model: "Model") -> "Variable":
        """
        Computes the next variable to be fixed. Should return None if all the
        integer variables have been fixed.
        :param model:
        :return: the next variable to be fixed.
        """
        ...

    @abstractmethod
    def backtrack(self, model: "Model"):
        """
        Indicates that the search has been backtracked. Required so that the
        search continues correctly after backtracking.
        :param model:
        """
        ...


class RandomFixingOrder(FixingOrderStrategy):
    """
    This shit is kinda hard to do... I don't really know a good way to do this.
    """
    name: str = 'RandomFixingOrder'

    _size: int
    _current_index: int
    _indices: List[int]

    def __init__(self, model: "Model"):
        self._indices = []
        binary_variables = []
        integer_variables = []
        var: Variable
        for var in model.variables:
            if var.type == VarType.BINARY:
                binary_variables.append(var.id)
            elif var.type == VarType.INTEGER:
                integer_variables.append(var.id)
        binary_variables = [binary_variables[i] for i in torch.randperm(len(binary_variables))]
        integer_variables = [integer_variables[i] for i in torch.randperm(len(integer_variables))]
        self._indices = binary_variables + integer_variables
        self._size = len(self._indices)
        self._current_index = 0

    def select_variable(self, model: "Model") -> Variable:
        # previous_index: int = self._current_index
        # previous_increment: int = self._last_increment
        # while self._current_index < self._size:
        #     var: "Variable" = architecture.get_var(self._current_index)
        #     if var.lb == var.ub:
        #         self._current_index += 1
        #     else:
        #         return var
        # self._current_index = previous_index
        # return None
        i: int = 0
        while i < self._size:
            var_id = self._indices[i]
            var: Variable = model.get_var(var_id)
            if var.lb != var.ub:
                return var
            i += 1
        return None

    def backtrack(self, model: "Model"):
        # self._current_index -= self._last_increment
        ...


class LeftRightOrder(FixingOrderStrategy):
    """
    In the original FPR paper, LR and type are different. However, we sort
    variables by type when we instantiate a Model instance using Gurobi. S0
    in that case, they're equivalent here.
    """
    name: str = "LeftRightOrder"

    _current_index: int
    _indices: List[int]
    _size: int

    def __init__(self, model: "Model"):
        self._indices = []
        var: Variable
        for var in model.variables:
            if var.type == VarType.BINARY or var.type == VarType.INTEGER:
                self._indices.append(var.id)
        self._size = len(self._indices)
        self._current_index = 0

    def select_variable(self, model: "Model") -> "Variable":
        # previous_index: int = self._current_index
        # previous_increment: int = self._last_increment
        # self._last_increment = 0
        # while self._current_index < self._size:
        #     self._last_increment += 1
        #     var: "Variable" = architecture.get_var(self._current_index)
        #     if var.lb == var.ub:
        #         self._current_index += 1
        #     else:
        #         return var
        # self._current_index = previous_index
        # self._last_increment = previous_increment
        # return None
        i: int = 0
        while i < self._size:
            var_id = self._indices[i]
            var: Variable = model.get_var(var_id)
            if var.lb != var.ub:
                return var
            i += 1
        return None

    def backtrack(self, model: "Model"):
        # self._current_index -= self._last_increment
        ...
