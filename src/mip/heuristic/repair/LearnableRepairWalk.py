import torch

from .RepairWalk import RepairWalk

from src.rl.architecture import MultilayerPerceptron
from src.rl.utils import ActionType, ActionHistory


class LearnableRepairWalk(RepairWalk):
    name: str = 'LearnableRepairWalk'

    # parameters
    _sample_indices: bool

    # architecture stuff
    _var_scoring_function: MultilayerPerceptron
    _cons_scoring_function: MultilayerPerceptron

    _action_history: ActionHistory

    def __init__(self,
                 params: "RepairWalkParams",
                 cons_scoring_function: MultilayerPerceptron,
                 var_scoring_function: MultilayerPerceptron,
                 sample_indices: bool = True,
                 **kwargs):
        super().__init__(params, **kwargs)
        self._var_scoring_function = var_scoring_function
        self._cons_scoring_function = cons_scoring_function
        self._sample_indices = sample_indices
        self._action_history = None

    def _sample_violated_constraint(self, model: "EnhancedModel") -> "Constraint":
        cons_ids = []
        features = []
        for cons in model.constraints:
            if cons.is_violated():
                cons_ids.append(cons.id)
                features.append(model.cons_features[cons.id])
        if len(cons_ids) == 0:
            return None
        features = torch.stack(features)
        scores = self._cons_scoring_function(features)
        if self._sample_indices:
            probabilities = torch.softmax(scores, dim=0)
            cons_idx = cons_ids[torch.multinomial(probabilities.T, 1).item()]
        else:
            argmax = torch.argmax(scores)
            cons_idx = cons_ids[argmax]
        return model.get_constraint(cons_idx)

    def _sample_var_candidate(self, model, cons, candidates):
        features = []
        cons_features = model.cons_features[cons.id]
        for var, new_domain, shift_damage in candidates:
            features.append(
                self._create_shift_candidate_feature(model, cons.id, cons_features, var, new_domain, shift_damage))
        features = torch.stack(features)
        scores = self._var_scoring_function(features)
        if self._sample_indices:
            probabilities = torch.softmax(scores, dim=0)
            idx = torch.multinomial(probabilities.T, 1).item()
        else:
            idx = torch.argmax(scores)
        var, new_domain, _ = candidates[idx]
        if self._action_history is not None:
            self._action_history.add((cons.id, var.id, new_domain), ActionType.REPAIR_VAR_SELECT)
        return var, new_domain

    def _create_shift_candidate_feature(self, model, cons_id, cons_features, var, domain_change, shift_damage):
        var_features = model.var_features[var.id]
        edge = model.graph.edges[(var.id, cons_id)]
        feat = torch.cat((cons_features,
                          var_features,
                          edge.features,
                          torch.Tensor([shift_damage])))
        return feat

    def _pick_candidate_greedily(self, cons, shift_candidates):
        var, new_domain, _ = min(shift_candidates, key=lambda t: t[2])
        if self._action_history is not None:
            self._action_history.add((cons.id, var.id, new_domain), ActionType.REPAIR_GREEDY)
        return var, new_domain
