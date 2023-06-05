import torch

import random

from src.rl.graph import Graph
from src.rl.params import GnnParams
from src.rl.model import MultilayerPerceptron
from src.mip.model import Model, VarType

from src.mip.heuristic.repair import RepairStrategy
from src.mip.propagation import Propagator
from src.rl.model import GraphNeuralNetwork
from src.mip.heuristic import FixPropRepair

from .FprNode import FprNode
from .FixingOrderStrategy import FixingOrderStrategy
from .ValueFixingStrategy import ValueFixingStrategy


class _FprlFixingOrderStrategy(FixingOrderStrategy):
    name: str = 'FprlFixingOrderStrategy'

    _fprl: "FixPropRepairLearn"
    _scoring_function: MultilayerPerceptron
    _sample_index: bool

    def __init__(self,
                 fprl: "FixPropRepairLearn",
                 scoring_function: MultilayerPerceptron,
                 sample_index: bool = True):
        self._fprl = fprl
        self._scoring_function = scoring_function
        self._sample_index = sample_index

    def select_variable(self, model: "Model") -> "Variable":
        var_ids = []
        features = []
        for var in model.variables:
            idx = var.id
            if var.type == VarType.CONTINUOUS:
                continue
            if var.lb != var.ub:
                var_ids.append(var.id)
                features.append(self._fprl._var_features[idx])
        if len(var_ids) == 0:
            return None
        features = torch.stack(features)
        scores = self._scoring_function(features)
        if self._sample_index:
            probabilities = torch.softmax(scores, dim=0)
            var_idx = var_ids[torch.multinomial(probabilities.T, 1).item()]
        else:
            idx = torch.argmax(scores)
            var_idx = var_ids[idx]
        return model.get_var(var_idx)

    def backtrack(self, model: "Model"):
        pass  # do nothing


class _FprlValueSelectionStrategy(ValueFixingStrategy):
    name: str = 'FprlValueFixingStrategy'

    _fprl: "FixPropRepairLearn"
    _scoring_function: MultilayerPerceptron
    _sample_index: bool

    def __init__(self,
                 fprl: "FixPropRepairLearn",
                 scoring_function: MultilayerPerceptron,
                 sample_index: bool = True):
        self._fprl = fprl
        self._scoring_function = scoring_function
        self._sample_index = sample_index

    def select_fixing_value(self, model: "Model", var: "Variable") -> tuple[int, int]:
        idx = var.id
        features = self._fprl._var_features[idx]
        score = self._scoring_function(features)
        p = torch.sigmoid(score).item()

        local_domain: "Domain" = var.local_domain
        lower_bound: int = int(local_domain.lower_bound)
        upper_bound: int = int(local_domain.upper_bound)

        if self._sample_index:
            lb_first = random.random() <= p
        else:
            lb_first = p < 0.5

        if lb_first:
            left_value, right_value = lower_bound, upper_bound
        else:
            left_value, right_value = upper_bound, lower_bound
        return left_value, right_value


class FixPropRepairLearn(FixPropRepair):
    _instance_graph: Graph
    _gnn: GraphNeuralNetwork
    _var_features: list[torch.Tensor]
    _cons_features: list[torch.Tensor]
    _initialized: bool

    def __init__(self,
                 gnn: GraphNeuralNetwork,
                 repair_strategy: RepairStrategy,
                 propagator: Propagator,
                 max_absolute_value: float = float('inf'),
                 propagate_fixings: bool = True,
                 repair: bool = True,
                 backtrack_on_infeasibility: bool = True):
        fixing_order_mlp = MultilayerPerceptron([GnnParams.intermediate_layers,
                                                 2 * GnnParams.intermediate_layers,
                                                 1])
        fixing_order_strategy = _FprlFixingOrderStrategy(self,
                                                         fixing_order_mlp)
        value_fixing_mlp = MultilayerPerceptron([GnnParams.intermediate_layers,
                                                 2 * GnnParams.intermediate_layers,
                                                 1])
        value_fixing_strategy = _FprlValueSelectionStrategy(self,
                                                            value_fixing_mlp)
        super().__init__(fixing_order_strategy,
                         value_fixing_strategy,
                         repair_strategy,
                         propagator,
                         max_absolute_value,
                         propagate_fixings,
                         repair,
                         backtrack_on_infeasibility)
        self._gnn = gnn
        self._initialized = False
        self._repair_strategy._fprl = self

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
        self.update(model)

        while len(search_stack) > 0 and continue_dive:
            node: FprNode = search_stack[-1]
            if not node.visited:
                self.update(model)
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
                success = self.handle_continuous_variables(model)
                if success:
                    self._logger.info("Solution found")
                else:
                    self._logger.info("No solution found")
        else:
            self._logger.info('no feasible integer variable fixing found')
        self._initialized = False
        return success

    def update(self, model: Model):
        if not self._initialized:
            self._instance_graph = Graph(model)
        else:
            self._instance_graph.update()
        self._var_features, self._cons_features = self._gnn(self._instance_graph)
