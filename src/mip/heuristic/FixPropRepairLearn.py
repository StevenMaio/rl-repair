import logging
from typing import List, Tuple

import torch

from src.rl.architecture import MultilayerPerceptron, PolicyArchitecture
from src.mip.model import VarType

from src.rl.mip import EnhancedModel
from src.rl.utils import ActionHistory, ActionType

from src.mip.heuristic.repair import RepairStrategy, repair_strategy_from_config
from src.mip.propagation import Propagator, LinearConstraintPropagator
from src.mip.heuristic import FixPropRepair
from src.utils.config import POLICY_ARCHITECTURE_CONFIG, REPAIR_STRAT_CONFIG, PARAMS

from .FprNode import FprNode
from src.mip.heuristic.fixing.FixingOrderStrategy import FixingOrderStrategy
from src.mip.heuristic.value.ValueFixingStrategy import ValueFixingStrategy
from ..model import Variable, Column, DomainChange, Constraint


class _FprlFixingOrderStrategy(FixingOrderStrategy):

    name: str = 'FprlFixingOrderStrategy'

    _scoring_function: MultilayerPerceptron
    _sample_index: bool

    _logger: logging.Logger

    def __init__(self,
                 scoring_function: MultilayerPerceptron,
                 sample_index: bool = True):
        self._scoring_function = scoring_function
        self._sample_index = sample_index
        self._logger = logging.getLogger(__package__)

    def select_variable(self, model: "EnhancedModel", generator=None) -> "Variable":
        var_ids = []
        features = []
        for var in model.variables:
            idx = var.id
            if var.type == VarType.CONTINUOUS:
                continue
            if var.lb != var.ub:
                var_ids.append(var.id)
                features.append(model.var_features[idx])
        if len(var_ids) == 0:
            return None
        features = torch.stack(features)
        scores = self._scoring_function(features)
        if self._sample_index:
            probabilities = torch.softmax(scores, dim=0)
            idx = torch.multinomial(probabilities.T,
                                    1,
                                    generator=generator).item()
            var_idx = var_ids[idx]
            p = probabilities[idx]
            self._logger.debug('VAR_SAMPLED idx=%d p=%.4f', idx, p)
        else:
            idx = torch.argmax(scores)
            var_idx = var_ids[idx]
        return model.get_var(var_idx)

    def init(self, model, generator=None):
        ...


class _FprlValueSelectionStrategy(ValueFixingStrategy):
    name: str = 'FprlValueFixingStrategy'

    _scoring_function: MultilayerPerceptron
    _sample_index: bool

    def __init__(self,
                 scoring_function: MultilayerPerceptron,
                 sample_index: bool = True):
        self._scoring_function = scoring_function
        self._sample_index = sample_index

    def select_fixing_value(self, model: "EnhancedModel", var: "Variable", generator=None) -> Tuple[int, int]:
        idx = var.id
        features = model.var_features[idx]
        score = self._scoring_function(features)
        p = torch.sigmoid(score).item()

        local_domain: "Domain" = var.local_domain
        lower_bound: int = int(local_domain.lower_bound)
        upper_bound: int = int(local_domain.upper_bound)

        if self._sample_index:
            lb_first = torch.rand(1, generator=generator).item() <= p
        else:
            lb_first = p < 0.5

        if lb_first:
            left_value, right_value = lower_bound, upper_bound
        else:
            left_value, right_value = upper_bound, lower_bound
        return left_value, right_value

    def init(self, model, generator=None):
        ...


class FixPropRepairLearn(FixPropRepair):
    _policy_architecture: PolicyArchitecture
    _action_history: ActionHistory
    _in_training: bool

    _sample_indices: bool

    def __init__(self,
                 policy_architecture,
                 repair_strategy: RepairStrategy,
                 propagator: Propagator,
                 discount_factor: float = 0.999,
                 max_absolute_value: float = float('inf'),
                 propagate_fixings: bool = True,
                 repair: bool = True,
                 backtrack_on_infeasibility: bool = True,
                 max_backtracks: int = 2,
                 sample_indices: bool = True,
                 in_training: bool = False):
        fixing_order_architecture = policy_architecture.fixing_order_architecture
        value_fixing_architecture = policy_architecture.value_fixing_architecture
        fixing_order_strategy = _FprlFixingOrderStrategy(fixing_order_architecture,
                                                         sample_index=sample_indices)
        value_fixing_strategy = _FprlValueSelectionStrategy(value_fixing_architecture,
                                                            sample_index=sample_indices)
        super().__init__(fixing_order_strategy,
                         value_fixing_strategy,
                         repair_strategy,
                         propagator,
                         discount_factor,
                         max_absolute_value,
                         propagate_fixings,
                         repair,
                         backtrack_on_infeasibility,
                         max_backtracks)
        self._policy_architecture = policy_architecture
        self._action_history = ActionHistory(in_training)
        self._in_training = in_training
        self._sample_indices = sample_indices
        if in_training:
            repair_strategy._action_history = self._action_history

    def find_solution(self, model: EnhancedModel, solution_filename=None, generator=None):
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
        self._action_history.clear()
        root = FprNode()
        search_stack: List[FprNode] = [root]
        success: bool = False
        continue_dive: bool = True
        num_backtracks: int = 0
        self._reward = 1 / self._discount_factor

        while len(search_stack) > 0 and continue_dive:
            node: FprNode = search_stack[-1]
            if not node.visited:
                model.update()
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
                        if num_backtracks < self._max_backtracks:
                            self._logger.debug("BACKTRACKING node=%d depth=%d",
                                               node.id,
                                               node.depth)
                            self._action_history.add(None, ActionType.BACKTRACK)
                            model.apply_domain_changes(*node.domain_changes, undo=True)
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
                self._reward = 0
                self._logger.info("No solution found")
        else:
            self._reward = 0
            self._logger.info('no feasible integer variable fixing found')
        return success

    @property
    def policy_architecture(self) -> PolicyArchitecture:
        return self._policy_architecture

    @policy_architecture.setter
    def policy_architecture(self, new_value):
        self._policy_architecture = new_value
        # change the architecture of the components as well
        self._repair_strategy._var_scoring_function = self._policy_architecture.var_scoring_function
        self._repair_strategy._cons_scoring_function = self._policy_architecture.cons_scoring_function
        self._fixing_order_strategy._scoring_function = self._policy_architecture.fixing_order_architecture
        self._value_fixing_strategy._scoring_function = self._policy_architecture.value_fixing_architecture

    def _find_solution_helper_node_loop(self,
                                        model: EnhancedModel,
                                        head: FprNode,
                                        generator=None) -> bool:
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
                prop_changes: list[DomainChange] = []
                for idx, _ in column:
                    constraint: Constraint = model.get_constraint(idx)
                    self._propagator.propagate(model, constraint, prop_changes)
                    model.apply_domain_changes(*prop_changes)
                    head.domain_changes.extend(prop_changes)
                    prop_changes.clear()
                    infeasible |= constraint.is_violated()
                    if infeasible:
                        break
        if infeasible and self._repair:
            self._logger.debug("starting repair")
            repair_changes: list[DomainChange] = []
            success: bool = self._repair_strategy.repair_domain(model,
                                                                repair_changes,
                                                                generator=generator)
            self._action_history.add(success, ActionType.REPAIR_FINISHED)
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
            if left_val == next_var.lb:
                self._action_history.add((next_var.id, 0), ActionType.FIXING)
            else:
                self._action_history.add((next_var.id, 1), ActionType.FIXING)

            left_fixing: DomainChange = DomainChange.create_fixing(next_var, left_val)
            left = FprNode(head, next_var.id, left_fixing)
            head.left = left

            right_fixing: DomainChange = DomainChange.create_fixing(next_var, right_val)
            right = FprNode(head, next_var.id, right_fixing)
            head.right = right
            return False
        else:
            return not model.violated

    @property
    def action_history(self):
        return self._action_history

    @property
    def in_training(self):
        return self._in_training

    @in_training.setter
    def in_training(self, other):
        self._in_training = other
        self._action_history._in_training = other

    @property
    def sample_indices(self):
        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, new_value):
        self._sample_indices = new_value
        self._repair_strategy._sample_indices = new_value
        self._fixing_order_strategy._sample_indices = new_value
        self._value_fixing_strategy._sample_indices = new_value

    @staticmethod
    def from_config(config: dict):
        propagator = LinearConstraintPropagator()
        params = config[PARAMS]
        sample_indices = params['sample_indices']

        policy_architecture = PolicyArchitecture.from_config(config[POLICY_ARCHITECTURE_CONFIG])
        repair_strategy = repair_strategy_from_config(config[REPAIR_STRAT_CONFIG])
        repair_strategy.sample_indices = sample_indices
        repair_strategy.cons_scoring_function = policy_architecture.cons_scoring_function
        repair_strategy.var_scoring_function = policy_architecture.var_scoring_function

        return FixPropRepairLearn(policy_architecture,
                                  repair_strategy,
                                  propagator,
                                  **params
                                  )
