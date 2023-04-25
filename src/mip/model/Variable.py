"""
Variable class. Wraps around a gurobipy.Var instance.

TODO:
    - should we update the underlying gurobipy.Var instance?

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
    _gurobi_var: gurobipy.Var
    _column: Column

    def __init__(self,
                 var_id: int,
                 variable_type: VarType,
                 global_domain: Domain,
                 gp_var: gurobipy.Var):
        self._var_id = var_id
        self._variable_type = variable_type
        self._global_domain = global_domain
        self._local_domain = global_domain.copy()
        self._gurobi_var = gp_var
        self._column = Column(self)

    @property
    def column(self) -> Column:
        return self._column

    @property
    def id(self) -> int:
        return self._var_id

    @property
    def variable_type(self) -> VarType:
        return self._variable_type

    @property
    def local_domain(self) -> Domain:
        return self._local_domain

    @local_domain.setter
    def local_domain(self, new_value: Domain):
        self._local_domain = new_value
        # TODO: should we update the gurobi.Var stuff -- no?

    @property
    def global_domain(self) -> Domain:
        return self._global_domain

    @property
    def lb(self):
        return self._local_domain.lower_bound

    @property
    def ub(self):
        return self._local_domain.upper_bound

    @staticmethod
    def from_gurobi_var(var_id: int, gurobi_var: gurobipy.Var) -> "Variable":
        var_type = VarType.from_str(gurobi_var.vType)
        domain = Domain(gurobi_var.lb, gurobi_var.ub)
        variable = Variable(var_id, var_type, domain, gurobi_var)
        return variable
