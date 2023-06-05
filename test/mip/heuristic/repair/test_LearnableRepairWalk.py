import logging
import random
import torch

from unittest import TestCase

from src.mip.params import RepairWalkParams
from src.mip.propagation import LinearConstraintPropagator
from src.mip.heuristic.repair import LearnableRepairWalk
from src.mip.heuristic import FixPropRepairLearn

from src.rl.model import GraphNeuralNetwork, MultilayerPerceptron
from src.rl.params import GnnParams

from src.utils import initialize_logger, REPAIR_LEVEL

from src.mip.model import Model, VarType, Sense, DomainChange, Variable

initialize_logger(level=REPAIR_LEVEL)



class TestLearnableRepairWalk(TestCase):

    @classmethod
    def setUpClass(cls):
        cls._logger = logging.getLogger(__package__)

    def test_violated_constraint_selection(self):
        """
        Description of the model in scrap for June 5, 2023
        :return:
        """
        num_samples: int = 1_000
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)

        # create the model
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        z_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        z: Variable = model.get_var(z_id)

        model.add_constraint([x_id, y_id],
                             [1.0, -1.0],
                             1.0,
                             Sense.LE)
        model.add_constraint([x_id, y_id],
                             [3.0, 1.0],
                             3.0,
                             Sense.GE)
        model.add_constraint([x_id, y_id, z_id],
                             [1.0, 1.0, 1.0],
                             2.0,
                             Sense.GE)

        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        num_learned_node_features = GnnParams.intermediate_layers
        num_edge_features = GnnParams.num_edge_features

        cons_scoring_function = MultilayerPerceptron([num_learned_node_features,
                                                      64,
                                                      1])
        var_scoring_function = MultilayerPerceptron([2*num_learned_node_features + num_edge_features,
                                                     64,
                                                     1])

        repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                           cons_scoring_function,
                                           var_scoring_function)

        gnn = GraphNeuralNetwork(GnnParams)
        fprl = FixPropRepairLearn(gnn,
                                  repair_strat,
                                  LinearConstraintPropagator())

        x_change = DomainChange.create_fixing(x, x.lb)
        z_change = DomainChange.create_fixing(z, z.lb)
        model.apply_domain_changes(x_change, z_change)
        fprl.update(model)
        self.assertTrue(model.violated)

        # test that the first constraint is ignored
        samples = []
        for _ in range(num_samples):
            samples.append(repair_strat._sample_violated_constraint(model).id)
        freq = torch.zeros(3)
        for i in samples:
            freq[i] += 1.0

        freq /= num_samples
        self.assertEqual(0.0, freq[0])
        print(freq)

        # test out sampling a constraint from the first violated constraint
        cons_idx = 1
        cons = model.get_constraint(cons_idx)
        var, shift = repair_strat._select_shift_candidate(model, cons)
        self.assertEqual(x, var)

        cons_idx = 2
        cons = model.get_constraint(cons_idx)
        samples = []
        for _ in range(num_samples):
            var, shift = repair_strat._select_shift_candidate(model, cons)
            samples.append(var.id)
        freq = torch.zeros(3)
        for i in samples:
            freq[i] += 1.0
        freq /= num_samples
        print(freq)

    def test_policy_gradient_with_cons_score(self):
        """
        Attempt to train a policy gradient
        :return:
        """
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)

        # create the model
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        z_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        z: Variable = model.get_var(z_id)

        model.add_constraint([x_id, y_id],
                             [1.0, -1.0],
                             1.0,
                             Sense.LE)
        model.add_constraint([x_id, y_id],
                             [3.0, 1.0],
                             3.0,
                             Sense.GE)
        model.add_constraint([x_id, y_id, z_id],
                             [1.0, 1.0, 1.0],
                             2.0,
                             Sense.GE)

        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        num_learned_node_features = GnnParams.intermediate_layers
        num_edge_features = GnnParams.num_edge_features

        cons_scoring_function = MultilayerPerceptron([num_learned_node_features,
                                                      64,
                                                      1])
        var_scoring_function = MultilayerPerceptron([2*num_learned_node_features + num_edge_features,
                                                     64,
                                                     1])

        repair_strat = LearnableRepairWalk(RepairWalkParams(),
                                           cons_scoring_function,
                                           var_scoring_function)

        gnn = GraphNeuralNetwork(GnnParams)
        fprl = FixPropRepairLearn(gnn,
                                  repair_strat,
                                  LinearConstraintPropagator())

        x_change = DomainChange.create_fixing(x, x.lb)
        z_change = DomainChange.create_fixing(z, z.lb)
        model.apply_domain_changes(x_change, z_change)
        fprl.update(model)
        self.assertTrue(model.violated)

        # try to see if we can force the network to prefer cons 1
        num_samples: int = 100
        learning_rate: float = 0.5

        cons_idx = 2
        cons_feat = fprl._cons_features[cons_idx]
        other_idx = 1
        other_feat = fprl._cons_features[other_idx]

        features = torch.stack((cons_feat, other_feat))
        scores = cons_scoring_function(features)
        probabilities = torch.softmax(scores, dim=0)
        print('before "training"')
        print(probabilities)

        for i in range(num_samples):
            cons_scoring_function.zero_grad()
            scores = cons_scoring_function(features)
            probabilities = torch.softmax(scores, dim=0)
            out = torch.log(probabilities[0])
            out.backward(retain_graph=True)
            for param in cons_scoring_function.parameters():
                grad = param.grad
                with torch.no_grad():
                    torch.add(param, learning_rate*grad, out=param)

        print('after "training"')
        scores = cons_scoring_function(features)
        probabilities = torch.softmax(scores, dim=0)
        print(probabilities)
