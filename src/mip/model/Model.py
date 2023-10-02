"""
This architecture wraps around a gurobipy.Model instance. Also performs some sorting
on the variables so that the binary variables appear first (this makes life
easier when implementing Fix-Prop-Repair).

TODO:
    - [x] implement instantiation from a gurobipy.Model instance
Author: Steven Maio
"""
import logging
import gurobipy
from gurobipy import GRB

from .Objective import Objective, ObjSense
from .Variable import VarType, Variable
from .Constraint import Constraint, Sense
from .Row import Row
from .Column import Column
from .Domain import *

from src.utils.constants import MAX_ABSOLUTE_VALUE

from typing import List


class Model:
    _gp_model: gurobipy.Model
    _num_variables: int
    _num_binary_variables: int
    _num_integer_variables: int
    _num_continuous_variables: int
    _num_constraints: int
    _variables: List[Variable]
    _constraints: List[Constraint]
    _objective: Objective
    _initialized: bool
    _violated: bool
    _lp_solved: bool

    def __init__(self):
        self._num_variables = 0
        self._num_binary_variables = 0
        self._num_integer_variables = 0
        self._num_continuous_variables = 0
        self._num_constraints = 0
        self._variables = []
        self._constraints = []
        self._objective = None
        self._initialized = False
        self._violated = True
        self._lp_solved = False

    def add_var(self,
                variable_type: VarType = VarType.INTEGER,
                lower_bound: float = 0,
                upper_bound: float = MAX_ABSOLUTE_VALUE) -> int:
        domain: Domain
        if variable_type == VarType.BINARY:
            domain = Domain(0, 1)
        else:
            domain = Domain(lower_bound, upper_bound)
        var = Variable(self._num_variables,
                       variable_type,
                       domain)
        self._num_variables += 1
        self._variables.append(var)
        if variable_type == VarType.BINARY:
            self._num_binary_variables += 1
        elif variable_type == VarType.INTEGER:
            self._num_integer_variables += 1
        else:
            self._num_continuous_variables += 1
        return var.id

    def add_constraint(self,
                       var_indices: List[int],
                       coefficients: List[float],
                       rhs: float,
                       sense: Sense = Sense.LE) -> int:
        constraint = Constraint(self._num_constraints,
                                sense,
                                rhs)
        row: Row = constraint.row
        c_id: int = constraint.id
        min_activity: float = 0
        max_activity: float = 0
        idx: int
        coef: float
        # TODO: handle infinite bounds
        for idx, coef in zip(var_indices, coefficients):
            row.add_term(idx, coef)
            x: Variable = self.get_var(idx)
            x.column.add_term(c_id, coef)
            if coef > 0:
                min_activity += x.lb * coef
                max_activity += x.ub * coef
            else:
                min_activity += x.ub * coef
                max_activity += x.lb * coef
        constraint.min_activity = min_activity
        constraint.max_activity = max_activity
        self._constraints.append(constraint)
        self._num_constraints += 1
        return c_id

    def set_objective(self,
                      var_indices: List[int],
                      coefficients: List[float],
                      sense: ObjSense = ObjSense.MINIMIZE):
        objective = Objective(var_indices, coefficients, sense)
        self._objective = objective

    def apply_domain_changes(self,
                             *domain_changes: DomainChange,
                             undo: bool = False,
                             recompute_activities: bool = True):
        """
        Applies the domain changes in domain_changes. If recompute_activities
        is set to true, then the activities of the constraints affected by the
        domain changes are updated, and their propagated flag is set to false.
        :param domain_changes:
        :param undo:
        :param recompute_activities:
        """
        logger: logging.Logger = logging.getLogger(__package__)
        d: DomainChange
        violated: bool = False
        for d in domain_changes:
            var: Variable = self.get_var(d.var_id)
            prev_domain: Domain = d.previous_domain
            new_domain: Domain = d.new_domain
            if new_domain == prev_domain:
                continue
            if undo:
                var.local_domain = prev_domain
                logger.debug("UNDO_DOMAIN_CHANGE var=%d prev_domain=[%.2f, %.2f] new_domain=[%.2f, %.2f] undo=%d",
                             var.id,
                             new_domain.lower_bound,
                             new_domain.upper_bound,
                             prev_domain.lower_bound,
                             prev_domain.upper_bound,
                             undo)
            else:
                var.local_domain = new_domain
                logger.debug("DO_DOMAIN_CHANGE var=%d prev_domain=[%.2f, %.2f] new_domain=[%.2f, %.2f] undo=%d",
                             var.id,
                             prev_domain.lower_bound,
                             prev_domain.upper_bound,
                             new_domain.lower_bound,
                             new_domain.upper_bound,
                             undo)
            if recompute_activities:
                # TODO: handle infinite bounds
                ub_shift: float = new_domain.upper_bound - prev_domain.upper_bound
                lb_shift: float = new_domain.lower_bound - prev_domain.lower_bound
                if undo:
                    ub_shift *= -1
                    lb_shift *= -1
                column: Column = var.column
                i: int
                for constraint_index, coefficient in column:
                    constraint: Constraint = self.get_constraint(constraint_index)
                    constraint.propagated = False
                    if coefficient > 0:
                        constraint.min_activity += lb_shift * coefficient
                        constraint.max_activity += ub_shift * coefficient
                    else:
                        constraint.min_activity += ub_shift * coefficient
                        constraint.max_activity += lb_shift * coefficient
                    violated |= constraint.is_violated()
                    logger.debug("CONSTRAINT_CHANGE id=%d min_activity=%.2f max_activity=%.2f rhs=%.2f violated=%d coef=%.2f",
                                 constraint.id,
                                 constraint.min_activity,
                                 constraint.max_activity,
                                 constraint.rhs,
                                 constraint.is_violated(),
                                 coefficient)
        if recompute_activities:
            if undo:
                # in this case, I think we have to check all the constraints again
                violated = False
                c: Constraint
                for c in self._constraints:
                    violated |= c.is_violated()
                self._violated = violated
            else:
                self._violated |= violated

    def get_var(self, var_id: int) -> Variable:
        return self._variables[var_id]

    def get_constraint(self, constraint_id: int) -> Constraint:
        return self._constraints[constraint_id]

    def convert_ge_constraints(self):
        """
        Converts GE constraints into LE constraints.
        :return:
        """
        logger: logging.Logger = logging.getLogger(__package__)
        logger.info("converting GE constraints to LE constraints")
        constraint: Constraint
        for constraint in self._constraints:
            if constraint.sense != Sense.GE:
                continue
            logger.debug("converting constraint=%d", constraint.id)
            row: Row = constraint.row
            constraint._rhs = -constraint.rhs
            constraint._sense = Sense.LE
            i: int
            for var_id, coefficient in row:
                var: Variable = self.get_var(var_id)
                column: Column = var.column
                column.modify_term(constraint.id, -coefficient)
                row.modify_term(var_id, -coefficient)

    @property
    def num_variables(self) -> int:
        return self._num_variables

    @property
    def num_binary_variables(self) -> int:
        return self._num_binary_variables

    @property
    def num_integer_variables(self) -> int:
        return self._num_integer_variables

    @property
    def num_continuous_variables(self) -> int:
        return self._num_continuous_variables

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def violated(self) -> bool:
        return self._violated

    @violated.setter
    def violated(self, new_value: bool):
        self._violated = new_value

    @property
    def constraints(self) -> List[Constraint]:
        return self._constraints

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    def get_gurobi_model(self) -> gurobipy.Model:
        return self._gp_model

    def init(self):
        """
        Initializes the architecture: computes the activities of the constraints, and
        sets the values of coefficient_sign for integer values. Also converts
        all GE inequalities to LE inequalities.
        :return:
        """
        if self._initialized:
            return
        logger: logging.Logger = logging.getLogger(__package__)
        violated: bool = False
        constraint: Constraint
        for constraint in self._constraints:
            row: Row = constraint.row
            min_activity: float = 0
            max_activity: float = 0
            for index, coefficient in row:
                var: Variable = self.get_var(index)
                if coefficient > 0:
                    min_activity += var.lb * coefficient
                    max_activity += var.ub * coefficient
                    var.num_up_locks += 1
                else:
                    min_activity += var.ub * coefficient
                    max_activity += var.lb * coefficient
                    var.num_down_locks += 1
            constraint.min_activity = min_activity
            constraint.max_activity = max_activity
            violated |= constraint.is_violated()
            logger.debug("constraint=%d min_activity=%.2f max_activity=%.2f rhs=%.2f violated=%d",
                         constraint.id,
                         constraint.min_activity,
                         constraint.max_activity,
                         constraint.rhs,
                         constraint.is_violated())
        # set the objective coefficients for the variables
        if self._objective is not None:
            for index, coefficient in self._objective:
                var: Variable = self.get_var(index)
                var.objective_coefficient = coefficient
        self._initialized = True
        self._violated = violated

    @staticmethod
    def from_gurobi_model(gp_model: gurobipy.Model,
                          solve_lp: bool = False) -> "Model":
        """
        Creates a Model instance from a gurobipy.Model instance. The variables
        are sorted based on their variable type. Binary variables come first,
        then general integer variables and finally continuous variables.

        :param gp_model:
        :param solve_lp:
        :return: a Model instance wrapping gurobi_model
        """
        logger: logging.Logger = logging.getLogger(__package__)
        logger.info('creating architecture from gurobipy.Model')
        model = Model()
        model._gp_model = gp_model

        var_index: int
        gp_var: gurobipy.Var
        for var_index, gp_var in enumerate(gp_model.getVars()):
            var = Variable.from_gurobi_var(var_index, gp_var)
            model._variables.append(var)
            var_type = VarType.from_str(gp_var.vType)
            if var_type == VarType.BINARY:
                model._num_binary_variables += 1
            elif var_type == VarType.INTEGER:
                model._num_integer_variables += 1
            elif var_type == VarType.CONTINUOUS:
                model._num_continuous_variables += 1
        model._num_variables = len(model.variables)

        # initialize the constraint data
        violated: bool = False
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
                index: int = gp_var.index
                coefficient: float = gurobi_row.getCoeff(i)
                row.add_term(index, coefficient)
                var: Variable = model._variables[index]
                column: Column = var.column
                column.add_term(constraint_index, coefficient)
                if coefficient > 0:
                    min_activity += coefficient * var.lb
                    max_activity += coefficient * var.ub
                    var.num_up_locks += 1
                else:
                    min_activity += coefficient * var.ub
                    max_activity += coefficient * var.lb
                    var.num_down_locks += 1
            constraint.min_activity = min_activity
            constraint.max_activity = max_activity
            violated |= constraint.is_violated()
            constraint_index += 1
        model._num_constraints = constraint_index
        model._initialized = True
        model._violated = violated
        logger.info('num_vars=%d num_bin=%d num_int=%d num_cont=%d num_constrs=%d',
                    model._num_variables,
                    model._num_binary_variables,
                    model._num_integer_variables,
                    model._num_continuous_variables,
                    model._num_constraints)
        return model

    def reset(self):
        self._violated = False
        for var in self._variables:
            var.reset()
        for constraint in self._constraints:
            constraint.reset(self)
            self._violated |= constraint.is_violated()

    def solve_relaxation(self):
        if not self._lp_solved:
            relaxation: gurobipy.Model = self._gp_model.relax()
            relaxation.optimize()
            if relaxation.Status == GRB.OPTIMAL:
                self._lp_solved = True
                for idx, gp_var in enumerate(relaxation.getVars()):
                    self._variables[idx].relaxation_value = gp_var.X
            else:
                raise Exception("Error solving relaxation")

    def update(self):
        # this does nothing in the base class
        ...
