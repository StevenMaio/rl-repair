from enum import Enum, auto


class ObjSense(Enum):
    MAXIMIZE = auto()
    MINIMIZE = auto()


class Objective:
    _var_indices: list[int]
    _coefficients: list[float]
    _sense: ObjSense
    _size: int

    def __init__(self,
                 var_indices: list[int],
                 coefficients: list[float],
                 sense: ObjSense = ObjSense.MINIMIZE):
        assert len(var_indices) == len(coefficients)
        self._var_indices = var_indices
        self._coefficients = coefficients
        self._sense = sense
        self._size = len(self._var_indices)

    def get_var_index(self, index: int) -> int:
        return self._var_indices[index]

    def get_coefficient(self, index: int) -> float:
        return self._coefficients[index]

    @property
    def sense(self) -> ObjSense:
        return self._sense

    @property
    def size(self) -> int:
        return self._size

    def __iter__(self):
        return zip(self._var_indices, self._coefficients)
