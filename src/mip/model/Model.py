"""
This model wraps around a gurobipy.Model instance. Also performs some sorting
on the variables so that the binary variables appear first (this makes life
easier when implementing Fix-Prop-Repair).

TODO:
    - implement instantiation from a gurobipy.Model instance

Author: Steven Maio
"""
import logging
import gurobipy

from .Variable import Variable, VarType
from .Constraint import Constraint
from .Row import Row
from .Column import Column


class Model:
    _gurobi_model: gurobipy.Model
    _num_variables: int
    _num_binary_variables: int
    _num_integer_variables: int
    _num_continuous_variables: int
    _num_constraints: int
    _variables: list[Variable]
    _constraints: list[Constraint]

    def __init__(self):
        self._gurobi_model = None
        self._num_variables = 0
        self._num_binary_variables = 0
        self._num_integer_variables = 0
        self._num_continuous_variables = 0
        self._num_constraints = 0
        self._variables = []
        self._constraints = []

    def get_var(self, var_id: int) -> Variable:
        return self._variables[var_id]

    def get_constraint(self, constraint_id: int) -> Constraint:
        return self._constraints[constraint_id]

    @property
    def num_variables(self):
        return self._num_variables

    @property
    def num_binary_variables(self):
        return self._num_binary_variables

    @property
    def num_integer_variables(self):
        return self._num_integer_variables

    @property
    def num_continuous_variables(self):
        return self._num_continuous_variables

    @staticmethod
    def from_gurobi_model(gp_model: gurobipy.Model) -> "Model":
        """
        Creates a Model instance from a gurobipy.Model instance.

        :param gp_model:
        :return: a Model instance wrapping gurobi_model
        """
        logger: logging.Logger = logging.getLogger(__package__)
        logger.info('creating model from gurobipy.Model')
        model = Model()
        binary_variables: list[gurobipy.Var] = []
        integer_variables: list[gurobipy.Var] = []
        continuous_variables: list[gurobipy.Var] = []
        variable_index_map: dict[int, int] = {}  # maps gurobi variable ids to ones in our representation

        # first sort the variables by type
        gp_var: gurobipy.Var
        for gp_var in gp_model.getVars():
            var_type = VarType.from_str(gp_var.vType)
            if var_type == VarType.BINARY:
                binary_variables.append(gp_var)
            elif var_type == VarType.INTEGER:
                integer_variables.append(gp_var)
            elif var_type == VarType.CONTINUOUS:
                continuous_variables.append(gp_var)

        model._num_binary_variables = len(binary_variables)
        model._num_integer_variables = len(integer_variables)
        model._num_continuous_variables = len(continuous_variables)
        model._num_variables = sum([
            model._num_binary_variables,
            model._num_integer_variables,
            model._num_continuous_variables
        ])

        # initialize variable data
        var_index: int = 0
        gp_var: gurobipy.Var
        for gp_var in binary_variables:
            var = Variable.from_gurobi_var(var_index, gp_var)
            variable_index_map[gp_var.index] = var_index
            model._variables.append(var)
            var_index += 1
        for gp_var in integer_variables:
            var = Variable.from_gurobi_var(var_index, gp_var)
            variable_index_map[gp_var.index] = var_index
            model._variables.append(var)
            var_index += 1
        for gp_var in continuous_variables:
            var = Variable.from_gurobi_var(var_index, gp_var)
            variable_index_map[gp_var.index] = var_index
            model._variables.append(var)
            var_index += 1

        # initialize the constraint data
        constraint_index: int = 0
        gp_constr: gurobipy.Constr
        for gp_constr in gp_model.getConstrs():
            # initialize row and columns and compute activities
            constraint = Constraint.from_gurobi_constr(constraint_index, gp_constr)
            min_activity: float = 0
            max_activity: float = 0
            model._constraints.append(constraint)
            gurobi_row: gurobipy.LinExpr = gp_model.getRow(gp_constr)
            row: Row = constraint.row
            for i in range(gurobi_row.size()):
                gp_var: gurobipy.Var = gurobi_row.getVar(i)
                index: int = variable_index_map[gp_var.index]
                coefficient: float = gurobi_row.getCoeff(i)
                row.add_term(index, coefficient)
                var: Variable = model._variables[index]
                column: Column = var.column
                column.add_term(constraint_index, coefficient)
                if coefficient > 0:
                    min_activity += coefficient * var.lb
                    max_activity += coefficient * var.ub
                else:
                    min_activity += coefficient * var.ub
                    max_activity += coefficient * var.lb
            constraint.min_activity = min_activity
            constraint.max_activity = max_activity
            constraint_index += 1
        model._num_constraints = constraint_index
        logger.info('num_vars=%d num_bin=%d num_int=%d num_cont=%d num_constrs=%d',
                    model._num_variables,
                    model._num_binary_variables,
                    model._num_integer_variables,
                    model._num_continuous_variables,
                    model._num_constraints)
        return model
