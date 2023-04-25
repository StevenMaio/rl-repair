from typing import Tuple


class Variable:
    ...


class Column:
    _var: Variable
    _size: int
    _indices: list[int]
    _coefficients: list[float]

    def __init__(self, var: Variable):
        self._var = var
        self._size = 0
        self._indices = []
        self._coefficients = []

    def add_term(self, index, coefficient):
        self._indices.append(index)
        self._coefficients.append(coefficient)
        self._size += 1

    def remove_term(self, constraint_index):
        if constraint_index in self._indices:
            index = self._indices.index(constraint_index)
            self._indices.pop(index)
            self._coefficients.pop(index)
            self._size -= 1

    def get_term(self, index: int) -> Tuple[int, float]:
        return self._indices[index], self._coefficients[index]

    def get_constraint_index(self, index: int) -> int:
        return self._indices[index]

    def get_coefficient(self, index: int) -> float:
        return self._coefficients[index]

    def modify_term(self, constraint_index: int, coefficient: float):
        if constraint_index in self._indices:
            index = self._indices.index(constraint_index)
            self._coefficients[index] = coefficient

    @property
    def size(self) -> int:
        return self._size

