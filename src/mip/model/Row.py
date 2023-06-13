from typing import Tuple, List


class Row:
    _constraint: "Constraint"
    _size: int
    _indices: List[int]
    _coefficients: List[float]

    def __init__(self, constraint: "Constraint"):
        self._constraint = constraint
        self._size = 0
        self._indices = []
        self._coefficients = []

    def add_term(self, index, coefficient):
        self._indices.append(index)
        self._coefficients.append(coefficient)
        self._size += 1

    def remove_term(self, var_index):
        if var_index in self._indices:
            index = self._indices.index(var_index)
            self._indices.pop(index)
            self._coefficients.pop(index)
            self._size -= 1

    def get_var_index(self, index: int) -> int:
        return self._indices[index]

    def get_coefficient(self, index: int) -> float:
        return self._coefficients[index]

    def get_term(self, index: int) -> Tuple[int, float]:
        return self._indices[index], self._coefficients[index]

    def modify_term(self, var_index: int, coefficient: float):
        if var_index in self._indices:
            index = self._indices.index(var_index)
            self._coefficients[index] = coefficient

    def __iter__(self):
        return zip(self._indices, self._coefficients)

    @property
    def size(self) -> int:
        return self._size
