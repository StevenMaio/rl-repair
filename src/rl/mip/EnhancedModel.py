import logging
import gurobipy
import torch

from typing import List

from src.mip.model import Variable, VarType, Constraint, Column, Row, Model, Sense

from src.rl.architecture import GraphNeuralNetwork
from src.rl.graph import Graph


class EnhancedModel(Model):
    _instance_graph: Graph
    _gnn: GraphNeuralNetwork
    _var_features: List[torch.Tensor]
    _cons_features: List[torch.Tensor]
    _initialized: bool

    # fields which are used to update nodes
    _largest_cons_size: int

    def __init__(self, gnn: GraphNeuralNetwork = None):
        super().__init__()
        self._gnn = gnn
        self._instance_graph = None
        self._initialized = False
        self._var_features = []
        self._cons_features = []

    def init(self):
        if self._initialized:
            return
        else:
            super().init()
            largest_cons_size = max(map(lambda c: c.row.size, self.constraints))
            self._largest_cons_size = largest_cons_size
            self._instance_graph = Graph(self)
            self._initialized = True

    def update(self):
        """
        Update the current node representations and compute a graph convolution.
        :return:
        """
        self._instance_graph.update(self)
        self._var_features, self._cons_features = self._gnn(self._instance_graph)

    @property
    def gnn(self) -> GraphNeuralNetwork:
        return self._gnn

    @gnn.setter
    def gnn(self, new_value: GraphNeuralNetwork):
        self._gnn = new_value

    @property
    def var_features(self) -> List[torch.Tensor]:
        return self._var_features

    @property
    def cons_features(self) -> List[torch.Tensor]:
        return self._cons_features

    @property
    def graph(self):
        return self._instance_graph

    @staticmethod
    def from_gurobi_model(gp_model: gurobipy.Model,
                          solve_lp: bool = False,
                          gnn: GraphNeuralNetwork = None,
                          convert_ge_cons: bool = False):
        """
        Creates a Model instance from a gurobipy.Model instance. The variables
        are sorted based on their variable type. Binary variables come first,
        then general integer variables and finally continuous variables.

        :param convert_ge_cons:
        :param gnn:
        :param gp_model:
        :param solve_lp:
        :return: a Model instance wrapping gurobi_model
        """
        logger: logging.Logger = logging.getLogger(__package__)
        logger.debug('creating architecture from gurobipy.Model')
        model = EnhancedModel(gnn)
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
            constraint = Constraint.from_gurobi_constr(constraint_index,
                                                       gp_constr,
                                                       convert_ge_cons)
            convert_cons = (gp_constr.sense == '>') and convert_ge_cons
            min_activity: float = 0
            max_activity: float = 0
            model._constraints.append(constraint)
            gurobi_row: gurobipy.LinExpr = gp_model.getRow(gp_constr)
            row: Row = constraint.row
            for i in range(gurobi_row.size()):
                gp_var: gurobipy.Var = gurobi_row.getVar(i)
                index: int = gp_var.index
                coefficient: float = gurobi_row.getCoeff(i)
                if convert_cons:
                    coefficient = -coefficient
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
            violated |= constraint.is_violated()
            logger.debug('CONSTRAINT_INITIALIZED id=%d min_activity=%.2f max_activity=%.2f',
                         constraint.id,
                         constraint.min_activity,
                         constraint.max_activity)
            constraint_index += 1
        model._num_constraints = constraint_index
        largest_cons_size = max(map(lambda c: c.row.size, model.constraints))
        model._largest_cons_size = largest_cons_size
        model._instance_graph = Graph(model)
        model._initialized = True
        model._violated = violated
        logger.info('num_vars=%d num_bin=%d num_int=%d num_cont=%d num_constrs=%d',
                    model._num_variables,
                    model._num_binary_variables,
                    model._num_integer_variables,
                    model._num_continuous_variables,
                    model._num_constraints)
        return model

    @property
    def largest_cons_size(self):
        return self._largest_cons_size
