import gurobipy as gp
from gurobipy import GRB

import torch
import random

from src.mip.model import VarType
from .LearningAlgorithm import LearningAlgorithm
from ..architecture import PolicyArchitecture

from ..mip import EnhancedModel
from ..utils import TensorList

from src.rl.utils import ActionHistory, ActionType


class PolicyGradientMethod(LearningAlgorithm):
    _num_epochs: int
    _num_trajectories: int
    _learning_parameter: float

    _num_successes: int

    def __init__(self,
                 num_epochs: int,
                 num_trajectories: int,
                 learning_parameter):
        self._num_epochs = num_epochs
        self._num_trajectories = num_trajectories
        self._learning_parameter = learning_parameter
        self._num_successes = 0

    def train(self, fprl, instances):
        self._num_successes = 0
        for round in range(self._num_epochs):
            problem = random.choice(instances)
            self._train_instance_loop(fprl, problem)

    def _train_instance_loop(self, fprl, instance):
        policy_architecture = fprl.policy_architecture
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.read(instance, env)
        model = EnhancedModel.from_gurobi_model(gp_model,
                                                gnn=policy_architecture.gnn)
        for trajectory_num in range(self._num_trajectories):
            with torch.no_grad():
                fprl.find_solution(model)
            if fprl.reward > 0:
                history = fprl.action_history
                self._handle_gradient_estimate(instance, fprl, history, gradient_estimate)
        gradient_estimate.scale(self._learning_parameter)
        gradient_estimate.add_to_iterator(policy_architecture.parameters())

    def _handle_gradient_estimate(self,
                                  instance: "EnhancedModel",
                                  fprl,
                                  action_history,
                                  gradient_estimate):
        """
        Computes the gradient for the problem instance.

        TODO: this needs to be adjusted after we determine the actual policy
              gradient approach.
        """
        policy_architecture: "PolicyArchitecture" = fprl.policy_architecture
        instance.reset()
        discount = fprl.discount_factor
        for action, type in action_history:
            instance.update()
            policy_architecture.zero_grad()
            if type == ActionType.FIXING:
                self._compute_fixing_action_gradient(instance,
                                                     fprl,
                                                     action,
                                                     gradient_estimate,
                                                     discount)
            else:
                # TODO: compute the policy gradient estimate
                cons_id, var_id = action
            discount *= discount

    def _compute_fixing_action_gradient(self,
                                        instance: "EnhancedModel",
                                        fprl,
                                        action,
                                        gradient_estimate,
                                        discount):
        policy_architecture = fprl.policy_architecture
        fixed_idx, to_upper = action

        # recreate the score computation
        var_ids = []
        features = []
        for i, var in enumerate(instance.variables):
            idx = var.id
            if var.type == VarType.CONTINUOUS:
                continue
            if var.lb != var.ub:
                var_ids.append(var.id)
                features.append(instance.var_features[idx])
            if idx == fixed_idx:
                component_idx = i
        if len(var_ids) == 0:
            return None
        features = torch.stack(features)
        scores = policy_architecture.fixing_order_architecture(features)
        probabilities = torch.softmax(scores, dim=0)
        fixed_p = probabilities[component_idx]
        fixed_p.backward()
        self._add_scaled_params_grad_to_estim(policy_architecture.parameters(),
                                              gradient_estimate,
                                              discount)
        policy_architecture.gnn.zero_grad()

        # recreate the bound fixing
        var_features = instance.var_features[fixed_idx]
        p_upper = torch.sigmoid(policy_architecture.value_fixing_architecture(var_features))
        p_upper = to_upper * p_upper + (1 - to_upper) * (1 - p_upper)
        p_upper.backward()

        policy_architecture.gnn.zero_grad()
        policy_architecture.fixing_order_architecture.zero_grad()
        self._add_scaled_params_grad_to_estim(policy_architecture.parameters(),
                                              gradient_estimate,
                                              discount)

    def _add_scaled_params_grad_to_estim(self,
                                         params,
                                         gradient_estim,
                                         scale):
        grads = []
        for p in params:
            g = p.grad
            if g is not None:
                grads.append(scale * g)
        gradient_estim.add_from_iter(grads)
