"""
Classes representing the order in which variables are fixed. There is hopefully
a cool, sexy way to do this efficiently. But I'm not smart enough to see it.

TODO:
    - [ ] think of a more efficient way to select a variable

Author: Steven Maio
"""
from abc import ABC, abstractmethod

from src.utils import range_permutation


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

    _num_binary_vars: int
    _num_integer_vars: int
    _size: int
    _current_index: int
    _indices: list[int]

    def __init__(self, model: "Model"):
        self._num_binary_vars = model.num_binary_variables
        self._num_integer_vars = model.num_integer_variables
        binary_indices = range_permutation(self._num_binary_vars)
        integer_indices = range_permutation(self._num_integer_vars)
        self._indices = binary_indices + [self._num_binary_vars + n for n in integer_indices]
        binary_indices.clear()
        integer_indices.clear()
        self._size = self._num_binary_vars + self._num_integer_vars
        self._current_index = 0

    def select_variable(self, model: "Model") -> "Variable":
        # previous_index: int = self._current_index
        # previous_increment: int = self._last_increment
        # while self._current_index < self._size:
        #     var: "Variable" = model.get_var(self._current_index)
        #     if var.lb == var.ub:
        #         self._current_index += 1
        #     else:
        #         return var
        # self._current_index = previous_index
        # return None
        i: int = 0
        while i < self._size:
            var_id = self._indices[i]
            var: "Variable" = model.get_var(var_id)
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
    _size: int

    def __init__(self, model: "Model"):
        self._current_index = 0
        self._size = model.num_integer_variables + model.num_binary_variables

    def select_variable(self, model: "Model") -> "Variable":
        # previous_index: int = self._current_index
        # previous_increment: int = self._last_increment
        # self._last_increment = 0
        # while self._current_index < self._size:
        #     self._last_increment += 1
        #     var: "Variable" = model.get_var(self._current_index)
        #     if var.lb == var.ub:
        #         self._current_index += 1
        #     else:
        #         return var
        # self._current_index = previous_index
        # self._last_increment = previous_increment
        # return None
        var_id: int = 0
        while var_id < self._size:
            var: "Variable" = model.get_var(var_id)
            if var.lb != var.ub:
                return var
            var_id += 1
        return None

    def backtrack(self, model: "Model"):
        # self._current_index -= self._last_increment
        ...
