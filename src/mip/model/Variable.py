"""
Variable class. Wraps around a gurobipy.Var instance.

TODO:
    - should we update the underlying gurobipy.Var instance?
    - need to know the sign of the objective coefficient

Author: Steven Maio
"""
from .Domain import Domain
import gurobipy

from enum import Enum, auto
from .Column import Column


class VarType(Enum):
    BINARY = auto()
    INTEGER = auto()
    IMPLIED_INTEGER = auto()    # WHAT THE FUCK AM I DOING WITH THIS???
    CONTINUOUS = auto()

    @staticmethod
    def from_str(s: str) -> "VarType":
        if s == "B":
            return VarType.BINARY
        elif s == "I":
            return VarType.INTEGER
        elif s == "C":
            return VarType.CONTINUOUS
        else:
            raise Exception("Invalid string value")


class Variable:
    _var_id: id
    _variable_type: VarType
    _local_domain: Domain
    _global_domain: Domain
    _gp_var: gurobipy.Var
    _column: Column
    _objective_coefficient: float
    # TODO: the features which depend on the LP relaxation

    def __init__(self,
                 var_id: int,
                 variable_type: VarType,
                 global_domain: Domain):
        self._var_id = var_id
        self._variable_type = variable_type
        self._global_domain = global_domain
        self._local_domain = global_domain.copy()
        self._column = Column(self)
        self._positive_objective_coefficient = 0
        self._objective_coefficient = 0

    @property
    def column(self) -> Column:
        return self._column

    @property
    def id(self) -> int:
        return self._var_id

    @property
    def type(self) -> VarType:
        return self._variable_type

    @property
    def local_domain(self) -> Domain:
        return self._local_domain

    @local_domain.setter
    def local_domain(self, new_value: Domain):
        self._local_domain = new_value

    @property
    def global_domain(self) -> Domain:
        return self._global_domain

    @property
    def lb(self):
        return self._local_domain.lower_bound

    @property
    def ub(self):
        return self._local_domain.upper_bound

    @property
    def objective_coefficient(self) -> float:
        return self._objective_coefficient

    @objective_coefficient.setter
    def objective_coefficient(self, new_value: float):
        self._objective_coefficient = new_value

    def get_gurobi_var(self) -> gurobipy.Var:
        return self._gp_var

    @staticmethod
    def from_gurobi_var(var_id: int, gp_var: gurobipy.Var) -> "Variable":
        var_type = VarType.from_str(gp_var.vType)
        domain = Domain(gp_var.lb, gp_var.ub)
        variable = Variable(var_id, var_type, domain)
        variable._gp_var = gp_var
        variable._objective_coefficient = gp_var.obj
        return variable
