from unittest import TestCase

from src.mip.params import RepairWalkParams
from src.mip.heuristic import FixPropRepair
from src.mip.model import *
from src.mip.propagation import LinearConstraintPropagator
from src.mip.heuristic.repair.RepairStrategy import RepairStrategy
from src.mip.heuristic.repair.RepairWalk import RepairWalk
from src.mip.heuristic.FixingOrderStrategy import *
from src.mip.heuristic.ValueFixingStrategy import *

import random

import gurobipy as gp
from gurobipy import GRB


class FakeRepairStrategy(RepairStrategy):
    name: str = "FakeRepairStrategy"

    def repair_domain(self, model: "Model", repair_changes: list["DomainChange"]) -> bool:
        return False


class TestFixPropRepair(TestCase):

    def test_find_solution1(self):
        """
        Tests to see if a set covering constraint is propagated in FPR
        :return:
        """
        random.seed(1)
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.GE)
        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        # initialize the components of FPR
        prop = LinearConstraintPropagator()
        fixing_order_strategy = RandomFixingOrder(model)
        value_fixing_strategy = RandomValueFixing()
        repair_strategy = FakeRepairStrategy()

        fpr = FixPropRepair(fixing_order_strategy,
                            value_fixing_strategy,
                            repair_strategy,
                            prop)
        fpr.find_solution(model)

    def test_find_solution2(self):
        """
        Tests to see if a partition constraint is propagated in FPR
        :return:
        """
        random.seed(100000)
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        # initialize the components of FPR
        prop = LinearConstraintPropagator()
        fixing_order_strategy = RandomFixingOrder(model)
        value_fixing_strategy = RandomValueFixing()
        repair_strategy = FakeRepairStrategy()

        fpr = FixPropRepair(fixing_order_strategy,
                            value_fixing_strategy,
                            repair_strategy,
                            prop)
        fpr.find_solution(model)

    def test_find_solution3(self):
        """
        Tests to see if a packing constraint is praopagated
        :return:
        """
        random.seed(100000)
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.LE)
        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        # initialize the components of FPR
        prop = LinearConstraintPropagator()
        fixing_order_strategy = RandomFixingOrder(model)
        value_fixing_strategy = RandomValueFixing()
        repair_strategy = FakeRepairStrategy()

        fpr = FixPropRepair(fixing_order_strategy,
                            value_fixing_strategy,
                            repair_strategy,
                            prop)
        fpr.find_solution(model)

    def test_find_solution4(self):
        """
        This test checks to see if backtracking occurs. I don't think there is
        a good way to directly test what happens. So in this case, I think we
        can only really look at the log output.
        :return:
        """
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        z_id: int = model.add_var(variable_type=VarType.BINARY)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([x_id, z_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([y_id, z_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        # initialize the components of FPR
        prop = LinearConstraintPropagator()
        fixing_order_strategy = LeftRightOrder(model)
        value_fixing_strategy = UpperBoundFirst()
        repair_strategy = FakeRepairStrategy()

        fpr = FixPropRepair(fixing_order_strategy,
                            value_fixing_strategy,
                            repair_strategy,
                            prop)
        success: bool = fpr.find_solution(model)
        self.assertFalse(success)

    def test_find_sol_w_cts_vars(self):
        """
        Checks to see if continuous variables are handled correctly.
        """
        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        gp_model = gp.Model(env=env)

        x1 = gp_model.addVar(vtype=GRB.BINARY, name="x1")
        x2 = gp_model.addVar(vtype=GRB.BINARY, name="x2")
        x3 = gp_model.addVar(vtype=GRB.CONTINUOUS, name="x3", lb=0.0, ub=10.0)
        gp_model.setObjective(x3, sense=GRB.MAXIMIZE)

        gp_model.addConstr(-x1 - x2 == -1)
        gp_model.addConstr(x1 + x3 <= 1.5)
        gp_model.presolve()

        model: Model = Model.from_gurobi_model(gp_model)
        # initialize the components of FPR
        prop = LinearConstraintPropagator()
        fixing_order_strategy = LeftRightOrder(model)
        value_fixing_strategy = UpperBoundFirst()
        params = RepairWalkParams()
        repair_strategy = RepairWalk(params)

        domain_changes = [
            DomainChange(0, Domain(0, 0), Domain.singleton(0)),
            DomainChange(1, Domain(0, 0), Domain.singleton(0)),
        ]
        model.apply_domain_changes(*domain_changes)

        fpr = FixPropRepair(fixing_order_strategy,
                            value_fixing_strategy,
                            repair_strategy,
                            prop)
        success: bool = fpr.find_solution(model)
        self.assertTrue(success)

