import logging

import gurobipy
from gurobipy import GRB

from .FixingOrderStrategy import FixingOrderStrategy
from .ValueFixingStrategy import ValueFixingStrategy
from src.mip.heuristic.repair import RepairStrategy
from src.mip.model import DomainChange, Model, Variable, Column, Constraint
from src.mip.propagation import Propagator
from .FprNode import FprNode

from src.utils import initialize_logger

initialize_logger()


class FixPropRepair:
    # FPR configuration
    _propagator: Propagator
    _max_absolute_value: float
    _fixing_order_strategy: FixingOrderStrategy
    _value_fixing_strategy: ValueFixingStrategy
    _repair_strategy: RepairStrategy

    # FPR parameters
    _propagate_fixings: bool
    _repair: bool
    _backtrack_on_infeasibility: bool

    _logger: logging.Logger

    def __init__(self,
                 fixing_order_strategy: FixingOrderStrategy,
                 value_fixing_strategy: ValueFixingStrategy,
                 repair_strategy: RepairStrategy,
                 propagator: Propagator,
                 max_absolute_value: float = float('inf'),
                 propagate_fixings: bool = True,
                 repair: bool = True,
                 backtrack_on_infeasibility: bool = True):
        self._logger = logging.getLogger(__package__)
        self._logger.info('initializing FPR -- max_abs_val=%.2f fixing_order_strategy=%s '
                          'value_fixing_strategy=%s repair_strategy=%s',
                          max_absolute_value,
                          fixing_order_strategy.name,
                          value_fixing_strategy.name,
                          repair_strategy.name)
        self._propagator = propagator
        self._max_absolute_value = max_absolute_value
        self._fixing_order_strategy = fixing_order_strategy
        self._value_fixing_strategy = value_fixing_strategy
        self._repair_strategy = repair_strategy
        self._propagate_fixings = propagate_fixings
        self._repair = repair
        self._backtrack_on_infeasibility = backtrack_on_infeasibility

    def _find_solution_helper_node_loop(self,
                                        model: Model,
                                        head: FprNode) -> bool:
        """
        Helper method that contains the inner loop logic of FPR
        """
        head.visited = True
        infeasible: bool = False
        if head.depth > 0:
            model.apply_domain_changes(*head.domain_changes)
            infeasible = model.violated
            if not infeasible and self._propagate_fixings:
                var_id: int = head.fixed_var_id
                fixed_var: Variable = model.get_var(var_id)
                column: Column = fixed_var.column
                i: int
                propagation_changes: list[DomainChange] = []
                for i in range(column.size):
                    constraint: Constraint = model.get_constraint(column.get_constraint_index(i))
                    self._propagator.propagate(model, constraint, propagation_changes)

                # apply the deduced domain changes from propagation
                model.apply_domain_changes(*propagation_changes)
                head.domain_changes.extend(propagation_changes)
                propagation_changes.clear()
                infeasible = model.violated
        if infeasible and self._repair:
            self._logger.debug("starting repair")
            repair_changes: list[DomainChange] = []
            success: bool = self._repair_strategy.repair_domain(model, repair_changes)
            self._logger.debug("repair success=%d", success)
            if success:
                head.domain_changes.extend(
                    repair_changes)  # append the repair changes to the current node for backtracking
                repair_changes.clear()
                infeasible = False
        if infeasible and self._backtrack_on_infeasibility:
            return False
        next_var: Variable = self._fixing_order_strategy.select_variable(model)
        if next_var is not None:
            left_val, right_val = self._value_fixing_strategy.select_fixing_value(model, next_var)

            left_fixing: DomainChange = DomainChange.create_fixing(next_var, left_val)
            left = FprNode(head, next_var.id, left_fixing)
            head.left = left

            right_fixing: DomainChange = DomainChange.create_fixing(next_var, right_val)
            right = FprNode(head, next_var.id, right_fixing)
            head.right = right
            return False
        else:
            return not model.violated

    def find_solution(self, model: Model):
        """
        What is the branching strategy? Do they just move on to the next variable?
        It looks like we always move in one direction, i.e., we either fix to
        the upper bound or to the lower bound. Is that the branching strategy? One
        child fixes to the node ot the upper or lower bound, and the other child
        fixes the variable to the opposite bound?

        In the event that all integral variables have been fixed and the model
        has continuous variables, then the LP which results from the fixings
        will be solved. This requires model has been initialized from a gurobipy.Model
        instance. If this is not the case, then an exception will be raised.
        :param model:
        :return:
        """
        if not model.initialized:
            raise Exception("Model has not yet been initialized")
        root = FprNode()
        search_stack: list[FprNode] = [root]
        success: bool = False
        continue_dive: bool = True

        while len(search_stack) > 0 and continue_dive:
            node: FprNode = search_stack[-1]
            if not node.visited:
                success: bool = self._find_solution_helper_node_loop(model, node)
                if success:
                    continue_dive = False
                    continue
                left: FprNode = node.left
                if left is not None:
                    search_stack.append(left)
            else:
                right: FprNode = node.right
                if right is not None and not right.visited:
                    search_stack.append(node.right)
                else:
                    if node.depth > 0:
                        self._logger.debug("backtracking node=%d depth=%d",
                                           node.id,
                                           node.depth)
                        model.apply_domain_changes(*node.domain_changes, undo=True)
                        self._fixing_order_strategy.backtrack(model)
                    search_stack.pop()
        if success:
            self._logger.info("feasible integer variable fixing found")
            if model.num_continuous_variables > 0:
                # solve the resulting LP to find a solution
                gp_model: gurobipy.Model = model.get_gurobi_model()
                self._logger.info("Gurobi model found. Solving LP")
                var: Variable
                for var in model.variables:
                    gp_var: gurobipy.Var = var.get_gurobi_var()
                    gp_var.lb = var.lb
                    gp_var.ub = var.ub
                gp_model.optimize()
                status: int = gp_model.status
                success = (status == GRB.OPTIMAL)
                if success:
                    self._logger.info("Solution found")
                else:
                    self._logger.info("No solution found")
        else:
            self._logger.info('no feasible integer variable fixing found')
        return success