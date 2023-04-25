import gurobipy

from enum import Enum, auto
from .Row import Row


class Sense(Enum):
    LE = auto()
    GE = auto()
    EQ = auto()

    @staticmethod
    def from_str(s: str) -> "Sense":
        if s == "<":
            return Sense.LE
        elif s == "=":
            return Sense.EQ
        elif s == ">":
            return Sense.GE
        else:
            raise Exception("Invalid string value")


class Constraint:
    _constraint_id: int
    _min_activity: float
    _max_activity: float
    _rhs: float
    _sense: Sense
    _gp_constraint: gurobipy.Constr
    _row: Row
    _propagated: bool

    def __init__(self,
                 constraint_id: int,
                 sense: Sense,
                 rhs: float):
        self._constraint_id = constraint_id
        self._rhs = rhs
        self._sense = sense
        self._min_activity = 0
        self._max_activity = 0
        self._row = Row(self)
        self._propagated = False

    def is_violated(self) -> bool:
        if self._sense == Sense.EQ:
            return self._rhs < self._min_activity or self._rhs > self._max_activity
        elif self._sense == Sense.GE:
            return self._max_activity < self._rhs
        elif self._sense == Sense.LE:
            return self._min_activity > self._rhs
        else:
            raise Exception("Constraint sense is invalid")

    @property
    def row(self) -> Row:
        return self._row

    @property
    def min_activity(self) -> float:
        return self._min_activity

    @min_activity.setter
    def min_activity(self, new_value: float):
        self._min_activity = new_value

    @property
    def max_activity(self) -> float:
        return self._max_activity

    @max_activity.setter
    def max_activity(self, new_value: float):
        self._max_activity = new_value

    @property
    def rhs(self) -> float:
        return self._rhs

    @property
    def sense(self) -> Sense:
        return self._sense

    @property
    def id(self):
        return self._constraint_id

    @property
    def propagated(self):
        return self._propagated

    @propagated.setter
    def propagated(self, new_value: bool):
        self._propagated = new_value

    @staticmethod
    def from_gurobi_constr(constraint_id: int, gp_constr: gurobipy.Constr) -> "Constraint":
        rhs: float = gp_constr.rhs
        sense = Sense.from_str(gp_constr.sense)
        constraint = Constraint(constraint_id, sense, rhs)
        constraint._gp_constraint = gp_constr
        return constraint
