import logging

import gurobipy
from gurobipy import GRB

from .FixingOrderStrategy import FixingOrderStrategy
from .ValueFixingStrategy import ValueFixingStrategy
from src.mip.heuristic.repair import RepairStrategy
from src.mip.model import DomainChange, Model, Variable, Column, Constraint
from src.mip.propagation import Propagator
from .FprNode import FprNode

class FixPropRepair:
    _discount_factor: float
    _reward: float

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
    _max_backtracks: int

    _logger: logging.Logger

    def __init__(self,
                 fixing_order_strategy: FixingOrderStrategy,
                 value_fixing_strategy: ValueFixingStrategy,
                 repair_strategy: RepairStrategy,
                 propagator: Propagator,
                 discount_factor: float = 0.99,
                 max_absolute_value: float = float('inf'),
                 propagate_fixings: bool = True,
                 repair: bool = True,
                 backtrack_on_infeasibility: bool = True,
                 max_backtracks: int = 2):
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
        self._reward = 0
        self._discount_factor = discount_factor
        self._max_backtracks = max_backtracks

    def _find_solution_helper_node_loop(self, model: Model, head: FprNode, generator=None) -> bool:
        """
        Helper method that contains the inner loop logic of FPR
        :param generator:
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
            success: bool = self._repair_strategy.repair_domain(model,
                                                                repair_changes,
                                                                generator=generator)
            self._logger.debug("repair success=%d", success)
            self._reward *= pow(self._discount_factor, self._repair_strategy.num_moves)
            if success:
                head.domain_changes.extend(
                    repair_changes)  # append the repair changes to the current node for backtracking
                repair_changes.clear()
                infeasible = False
        if infeasible and self._backtrack_on_infeasibility:
            return False
        next_var: Variable = self._fixing_order_strategy.select_variable(model,
                                                                         generator=generator)
        if next_var is not None:
            left_val, right_val = self._value_fixing_strategy.select_fixing_value(model,
                                                                                  next_var,
                                                                                  generator=generator)

            left_fixing: DomainChange = DomainChange.create_fixing(next_var, left_val)
            left = FprNode(head, next_var.id, left_fixing)
            head.left = left

            right_fixing: DomainChange = DomainChange.create_fixing(next_var, right_val)
            right = FprNode(head, next_var.id, right_fixing)
            head.right = right
            return False
        else:
            return not model.violated

    def find_solution(self, model: Model, solution_filename=None, generator=None):
        """
        What is the branching strategy? Do they just move on to the next variable?
        It looks like we always move in one direction, i.e., we either fix to
        the upper bound or to the lower bound. Is that the branching strategy? One
        child fixes to the node ot the upper or lower bound, and the other child
        fixes the variable to the opposite bound?

        In the event that all integral variables have been fixed and the architecture
        has continuous variables, then the LP which results from the fixings
        will be solved. This requires architecture has been initialized from a gurobipy.Model
        instance. If this is not the case, then an exception will be raised.
        :param generator:
        :param solution_filename:
        :param model:
        :return:
        """
        if not model.initialized:
            raise Exception("Model has not yet been initialized")
        root = FprNode()
        search_stack: list[FprNode] = [root]
        success: bool = False
        continue_dive: bool = True
        num_backtracks: int = 0
        self._reward = 1 / self._discount_factor

        while len(search_stack) > 0 and continue_dive:
            node: FprNode = search_stack[-1]
            if not node.visited:
                self._reward *= self._discount_factor
                success: bool = self._find_solution_helper_node_loop(model,
                                                                     node,
                                                                     generator=generator)
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
                        if num_backtracks < self._max_backtracks:
                            model.apply_domain_changes(*node.domain_changes, undo=True)
                            self._fixing_order_strategy.backtrack(model)
                            num_backtracks += 1
                        else:
                            success = False
                            break
                    search_stack.pop()
        if success:
            success = self.determine_feasibility(model)
            if success:
                self._logger.info("Solution found")
                if solution_filename is not None:
                    model.get_gurobi_model().write(solution_filename)
            else:
                self._logger.info("No solution found")
                self._reward = 0
        else:
            self._logger.info('no feasible integer variable fixing found')
            self._reward = 0
        return success

    def determine_feasibility(self, model: Model) -> bool:
        # solve the resulting LP to find a solution
        gp_model: gurobipy.Model = model.get_gurobi_model()
        self._logger.info("Gurobi architecture found. Solving LP")
        var: Variable
        for var in model.variables:
            gp_var: gurobipy.Var = var.get_gurobi_var()
            gp_var.lb = var.lb
            gp_var.ub = var.ub
        gp_model.optimize()
        status: int = gp_model.status
        success = (status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL)
        return success

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def propagator(self) -> Propagator:
        return self._propagator

    @property
    def repair_strategy(self) -> RepairStrategy:
        return self._repair_strategy
