from unittest import TestCase

from src.rl.architecture import GraphNeuralNetwork
from src.rl.params import GnnParams
from src.rl.mip import EnhancedModel

from src.mip.propagation import LinearConstraintPropagator
from src.mip.model import VarType, Sense, Model, Variable, DomainChange
from src.mip.heuristic.repair import RepairStrategy

from src.mip.heuristic import FixPropRepairLearn

import torch

RNG_SEED = 1


class FakeRepairStrategy(RepairStrategy):
    name: str = "FakeRepairStrategy"

    def repair_domain(self, model: "Model", repair_changes: list["DomainChange"]) -> bool:
        return False


class TestFixPropRepairLearn(TestCase):
    _model: EnhancedModel

    def setUp(self):
        torch.manual_seed(RNG_SEED)
        model = EnhancedModel()

        x = model.add_var(variable_type=VarType.BINARY)
        y = model.add_var(variable_type=VarType.BINARY)
        z = model.add_var(variable_type=VarType.BINARY)
        u = model.add_var(variable_type=VarType.INTEGER,
                          upper_bound=10)

        model.add_constraint([x, y],
                             [1.0, 1.0],
                             1.0,
                             sense=Sense.GE)
        model.add_constraint([x, z],
                             [1.0, 1.0],
                             1.0,
                             sense=Sense.GE)
        model.add_constraint([x, y, z, u],
                             [-5.0, 1.0, 3.0, -0.5],
                             1.0,
                             sense=Sense.LE)
        model.convert_ge_constraints()
        model.init()
        self._model = model

    def test_fixing_order_function(self):
        num_samples = 1_000
        model = self._model
        gnn = GraphNeuralNetwork(GnnParams)
        model.gnn = gnn
        fprl = FixPropRepairLearn(FakeRepairStrategy(),
                                  LinearConstraintPropagator())
        model.update()

        fixing_order_strategy = fprl._fixing_order_strategy
        samples = []
        for _ in range(num_samples):
            samples.append(fixing_order_strategy.select_variable(model).id)
        freq = torch.zeros(4)

        for i in samples:
            freq[i] += 1.0

        freq /= num_samples
        fixing_order_mlp = fixing_order_strategy._scoring_function
        var_features = torch.stack(model.var_features)
        scores = fixing_order_mlp(var_features)
        probabilities = torch.softmax(scores, dim=0)

        print(freq)
        print(probabilities)
        print(torch.abs(freq - probabilities).sum())

    def test_fixing_order_function2(self):
        """
        Test that the fixing order selection skips variables which have already
        been fixed.
        :return:
        """
        num_samples = 1_000
        model = self._model
        gnn = GraphNeuralNetwork(GnnParams)
        model.gnn = gnn
        fprl = FixPropRepairLearn(FakeRepairStrategy(),
                                  LinearConstraintPropagator())
        # fix the first variable
        var_idx = 1
        x: Variable = model.get_var(var_idx)
        x_bound_change = DomainChange.create_fixing(x, x.lb)
        model.apply_domain_changes(x_bound_change)
        model.update()

        fixing_order_strategy = fprl._fixing_order_strategy
        samples = []
        for _ in range(num_samples):
            samples.append(fixing_order_strategy.select_variable(model).id)
        freq = torch.zeros(4)

        for i in samples:
            freq[i] += 1.0

        freq /= num_samples
        print(freq)
        self.assertEqual(freq[var_idx], 0)

    def test_value_fixing_strategy(self):
        num_samples = 10_000

        model = self._model
        gnn = GraphNeuralNetwork(GnnParams)
        model.gnn = gnn
        fprl = FixPropRepairLearn(FakeRepairStrategy(),
                                  LinearConstraintPropagator())
        model.update()

        fixing_order_strategy = fprl._fixing_order_strategy
        value_fixing_strategy = fprl._value_fixing_strategy

        var = fixing_order_strategy.select_variable(model)

        vls_mlp = value_fixing_strategy._scoring_function
        score = vls_mlp(model.var_features[var.id])
        p = torch.sigmoid(score).item()

        samples = []
        lb = int(var.lb)
        for _ in range(num_samples):
            first, _ = value_fixing_strategy.select_fixing_value(model, var)
            if first == lb:
                samples.append(1)
            else:
                samples.append(0)

        print(sum(samples) / num_samples, p)
