import gurobipy
import torch

from .RepairWalk import RepairWalk

from src.rl.architecture import MultilayerPerceptron


class LearnableRepairWalk(RepairWalk):
    name: str = 'LearnableRepairWalk'

    # parameters
    _sample_indices: bool

    # architecture stuff
    _var_scoring_function: MultilayerPerceptron
    _cons_scoring_function: MultilayerPerceptron

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
        for var, domain_change, shift_damage in candidates:
            features.append(self._create_shift_candidate_feature(model,
                                                                 cons_features,
                                                                 var,
                                                                 domain_change,
                                                                 shift_damage))
        features = torch.stack(features)
        scores = self._var_scoring_function(features)
        if self._sample_indices:
            probabilities = torch.softmax(scores, dim=0)
            idx = torch.multinomial(probabilities.T, 1).item()
        else:
            idx = torch.argmax(scores)
        var, domain_change, _ = candidates[idx]
        return var, domain_change

    def _create_shift_candidate_feature(self, model, cons_features, var, domain_change, shift_damage):
        var_features = model.var_features[var.id]
        feat = torch.cat((cons_features, var_features, torch.Tensor([shift_damage])))
        return feat
