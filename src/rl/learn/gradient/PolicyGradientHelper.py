"""
Contains the helper methods for computing the policy gradient estimate. This
computation involves simulating the agent, as we need everything to be in
exact same state, at which the decision was made.
"""
import torch
import logging

from src.mip.heuristic import FixPropRepairLearn
from src.mip.model import VarType, DomainChange
from rl.mip import EnhancedModel
from rl.utils import TensorList, ActionType


class PolicyGradientHelper:
    _fprl: FixPropRepairLearn
    _model: EnhancedModel
    _logger: logging.Logger

    def __init__(self, fprl, model):
        self._fprl = fprl
        self._model = model
        self._logger = logging.getLogger(__package__)

    def compute_gradient_estimate(self, reset_model=True):
        policy_architecture = self._fprl.policy_architecture
        policy_architecture.zero_grad()
        history = self._fprl.action_history
        gradient_estimate = TensorList.zeros_like(policy_architecture.parameters())
        discount_factor = self._fprl.discount_factor
        discounted_reward = self._fprl.reward * pow(discount_factor, len(history) - 1)
        domain_changes = []
        for data, action_type in history:
            self._logger.debug('PROCESSING_ACTION type=%s', action_type)
            self._model.update()
            policy_architecture.zero_grad()
            if action_type == ActionType.FIXING:
                self._process_fixing_action(data,
                                            discounted_reward,
                                            gradient_estimate,
                                            domain_changes)
                discounted_reward /= discount_factor
            elif action_type == ActionType.BACKTRACK:
                last_changes = domain_changes.pop()
                self._model.apply_domain_changes(*last_changes, undo=True)
            elif action_type == ActionType.REPAIR_FINISHED:
                self._process_repair_finished(data, domain_changes)
            elif action_type == ActionType.REPAIR_GREEDY:
                self._process_greedy_repair_action(data,
                                                   discounted_reward,
                                                   gradient_estimate,
                                                   domain_changes)
                discounted_reward /= discount_factor
            else:
                self._process_policy_repair_action(data,
                                                   discounted_reward,
                                                   gradient_estimate,
                                                   domain_changes)
                discounted_reward /= discount_factor
        if reset_model:
            self._model.reset()
        return gradient_estimate

    def _process_fixing_action(self,
                               data,
                               discounted_reward,
                               gradient_estimate,
                               domain_changes):
        policy_architecture = self._fprl.policy_architecture
        var_id, bound = data

        var = self._model.get_var(var_id)
        var_features = self._model.var_features[var_id]

        var_scorer = policy_architecture.fixing_order_architecture
        bound_scorer = policy_architecture.value_fixing_architecture

        var_prob = self._get_fix_order_prob(var_scorer, self._model, var_id)
        to_upper_prob = torch.sigmoid(bound_scorer(var_features))

        if bound == 0:
            p = var_prob * (1 - to_upper_prob)
            fixed_value = var.lb
        else:
            p = var_prob * to_upper_prob
            fixed_value = var.ub

        p.backward(retain_graph=True)
        self._update_gradient_estimate(policy_architecture,
                                       gradient_estimate,
                                       discounted_reward)

        current_changes = [DomainChange.create_fixing(var, fixed_value)]
        self._model.apply_domain_changes(*current_changes)
        infeasible = self._model.violated
        if not infeasible:
            propagator = self._fprl.propagator

            prop_changes = []
            col = var.column
            for idx, _ in col:
                cons = self._model.get_constraint(idx)
                propagator.propagate(self._model, cons, prop_changes)
                self._model.apply_domain_changes(*prop_changes)
                current_changes.extend(prop_changes)
                prop_changes.clear()
                infeasible |= cons.is_violated()
                if infeasible:
                    break
        domain_changes.append(current_changes)

    def _process_greedy_repair_action(self,
                                      data,
                                      discounted_reward,
                                      gradient_estimate,
                                      domain_changes):
        policy_architecture = self._fprl.policy_architecture
        cons_id, var_id, new_domain = data
        var = self._model.get_var(var_id)

        cons_scorer = policy_architecture.cons_scoring_function
        cons_prob = self._get_repair_cons_prob(cons_scorer, self._model, cons_id)

        cons_prob.backward(retain_graph=True)
        self._update_gradient_estimate(policy_architecture,
                                       gradient_estimate,
                                       discounted_reward)
        domain_chg = DomainChange(var_id, var.local_domain, new_domain)
        self._model.apply_domain_changes(domain_chg)
        domain_changes[-1].append(domain_chg)

    def _process_policy_repair_action(self,
                                      data,
                                      discounted_reward,
                                      gradient_estimate,
                                      domain_changes):
        policy_architecture = self._fprl.policy_architecture
        cons_id, var_id, new_domain = data
        cons = self._model.get_constraint(cons_id)
        var = self._model.get_var(var_id)

        cons_scorer = policy_architecture.cons_scoring_function
        var_scorer = policy_architecture.var_scoring_function
        cons_prob = self._get_repair_cons_prob(cons_scorer, self._model, cons_id)
        var_prob = self._get_repair_var_prob(var_scorer, self._model, cons, var_id)

        p = cons_prob * var_prob
        p.backward(retain_graph=True)

        self._update_gradient_estimate(policy_architecture,
                                       gradient_estimate,
                                       discounted_reward)
        domain_chg = DomainChange(var_id, var.local_domain, new_domain)
        self._model.apply_domain_changes(domain_chg)
        domain_changes[-1].append(domain_chg)

    def _get_repair_var_prob(self,
                             var_scorer,
                             model,
                             cons,
                             var_id):
        repair_strat = self._fprl.repair_strategy
        shift_candidates, _ = repair_strat.find_shift_candidates(model,
                                                                 cons)
        features = []
        cons_features = model.cons_features[cons.id]
        for idx, (var, new_domain, shift_damage) in enumerate(shift_candidates):
            features.append(
                self._create_shift_candidate_feature(model,
                                                     cons.id,
                                                     cons_features,
                                                     var,
                                                     new_domain,
                                                     shift_damage))
            if var.id == var_id:
                policy_idx = idx
        features = torch.stack(features)
        scores = var_scorer(features)
        probabilities = torch.softmax(scores, dim=0)
        return probabilities[policy_idx]

    def _create_shift_candidate_feature(self,
                                        model,
                                        cons_id,
                                        cons_features,
                                        var,
                                        domain_change,
                                        shift_damage):
        var_features = model.var_features[var.id]
        edge = model.graph.edges[(var.id, cons_id)]
        feat = torch.cat((cons_features,
                          var_features,
                          edge.features,
                          torch.Tensor([shift_damage])))
        return feat

    @staticmethod
    def _get_repair_cons_prob(cons_scorer, model, cons_id):
        features = []
        num_violated_cons = 0
        policy_idx = -1
        for cons in model.constraints:
            if cons.is_violated():
                features.append(model.cons_features[cons.id])
                if cons.id == cons_id:
                    policy_idx = num_violated_cons
                num_violated_cons += 1
        features = torch.stack(features)
        scores = cons_scorer(features)
        probabilities = torch.softmax(scores, dim=0)
        return probabilities[policy_idx]

    @staticmethod
    def _get_fix_order_prob(var_scorer, model, var_id):
        features = []
        num_non_fixed_ints = 0
        policy_idx = -1
        for var in model.variables:
            if var.type == VarType.CONTINUOUS:
                continue
            idx = var.id
            if var.lb != var.ub:
                features.append(model.var_features[idx])
                if idx == var_id:
                    policy_idx = num_non_fixed_ints
                num_non_fixed_ints += 1
        features = torch.stack(features)
        scores = var_scorer(features)
        prob_vector = torch.softmax(scores, dim=0)
        return prob_vector[policy_idx]

    @staticmethod
    def _update_gradient_estimate(policy_architecture,
                                  gradient_estimate,
                                  discounted_reward):
        with torch.no_grad():
            for g, theta in zip(gradient_estimate,
                                policy_architecture.parameters()):
                grad = theta.grad
                if grad is not None:
                    g.add_(discounted_reward * grad)

    def _process_repair_finished(self, success, domain_changes):
        model = self._model
        if not success:
            changes_to_backtrack = domain_changes.pop()
            model.apply_domain_changes(changes_to_backtrack, undo=True)
        else:
            model.violated = False
