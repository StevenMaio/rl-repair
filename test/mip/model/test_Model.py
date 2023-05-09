import logging
from unittest import TestCase

import gurobipy as gp
from gurobipy import GRB

from src.mip.model import *
from src.utils import initialize_logger
from src.utils.constants import MAX_ABSOLUTE_VALUE

initialize_logger()


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls):
        # initialize model based on one from a Coursera course
        logger: logging.Logger = logging.getLogger(__package__)

        env = gp.Env()
        env.setParam(GRB.Param.OutputFlag, 0)
        m = gp.Model(env=env)
        x1 = m.addVar(vtype=GRB.INTEGER, name="x1")
        x2 = m.addVar(vtype=GRB.INTEGER, name="x2")
        x4 = m.addVar(vtype=GRB.CONTINUOUS, name="x4")
        m.addVar(vtype=GRB.BINARY, name="x3")
        m.addConstr(3 * x1 + 2 * x2 <= 6, name="c1")
        m.addConstr(-3 * x1 + 2 * x2 <= 0, name="c2")
        m.setObjective(x2, sense=GRB.MAXIMIZE)
        m.addConstr(x1 >= 0)
        m.addConstr(x2 >= 0)
        m.addConstr(x4 == 0)
        m.optimize()
        cls._instance = Model.from_gurobi_model(m)
        logger.info('class instance initialized')

    def test_from_gurobi_model(self):
        exception_raised: bool = False
        model: Model
        try:
            model = self._instance
        except AttributeError:
            exception_raised = True
        self.assertFalse(exception_raised)

        # expected values
        actual_num_variables: int = 4
        actual_num_binary_variables: int = 1
        actual_num_integer_variables: int = 2
        actual_num_continuous_variables: int = 1

        self.assertEqual(model.num_variables, actual_num_variables)
        self.assertEqual(model.num_binary_variables, actual_num_binary_variables)
        self.assertEqual(model.num_integer_variables, actual_num_integer_variables)
        self.assertEqual(model.num_continuous_variables, actual_num_continuous_variables)
        self.assertTrue(model.initialized)

    def test_get_var(self):
        model: Model = self._instance
        counter: int = 0

        # test column of variable 1
        x1: Variable = model.get_var(0)
        column: Column = x1.column
        actual_size: int = 3
        actual_indices: list[int] = [0, 1, 2]
        actual_coefficients: list[float] = [3.0, -3.0, 1.0]
        self.assertEqual(column.size, actual_size)
        self.assertListEqual(column._indices, actual_indices)
        self.assertListEqual(column._coefficients, actual_coefficients)
        self.assertEqual(0, x1.objective_coefficient)

        x2: Variable = model.get_var(1)
        self.assertEqual(1, x2.objective_coefficient)

    def test_get_constraint(self):
        model: Model = self._instance

        # expected values after variable sorting for constraint 0
        c: Constraint = model.get_constraint(0)
        row: Row = c.row
        actual_size: int = 2
        actual_rhs: float = 6.0
        actual_indices: list[int] = [0, 1]
        actual_coefficients: list[float] = [3.0, 2.0]
        self.assertEqual(row.size, actual_size)
        self.assertEqual(c.rhs, actual_rhs)
        self.assertEqual(c.sense, Sense.LE)
        self.assertListEqual(row._indices, actual_indices)
        self.assertListEqual(row._coefficients, actual_coefficients)

        # test constraint 1
        c: Constraint = model.get_constraint(1)
        row: Row = c.row
        actual_size: int = 2
        actual_rhs: float = 0
        actual_indices: list[int] = [0, 1]
        actual_coefficients: list[float] = [-3.0, 2.0]
        self.assertEqual(row.size, actual_size)
        self.assertEqual(c.rhs, actual_rhs)
        self.assertEqual(c.sense, Sense.LE)
        self.assertListEqual(row._indices, actual_indices)
        self.assertListEqual(row._coefficients, actual_coefficients)

        # test constraint 2
        c: Constraint = model.get_constraint(2)
        row: Row = c.row
        actual_size: int = 1
        actual_rhs: float = 0
        actual_indices: list[int] = [0]
        actual_coefficients: list[float] = [1]
        self.assertEqual(row.size, actual_size)
        self.assertEqual(c.rhs, actual_rhs)
        self.assertEqual(c.sense, Sense.GE)
        self.assertListEqual(row._indices, actual_indices)
        self.assertListEqual(row._coefficients, actual_coefficients)

        # test constraint 4
        c: Constraint = model.get_constraint(4)
        row: Row = c.row
        actual_size: int = 1
        actual_rhs: float = 0
        actual_indices: list[int] = [2]
        actual_coefficients: list[float] = [1]
        self.assertEqual(row.size, actual_size)
        self.assertEqual(c.rhs, actual_rhs)
        self.assertEqual(c.sense, Sense.EQ)
        self.assertListEqual(row._indices, actual_indices)
        self.assertListEqual(row._coefficients, actual_coefficients)

    def test_add_var(self):
        model = Model()
        model.add_var()
        model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(0)
        y: Variable = model.get_var(1)

        # test some asserts
        x_id: int = 0
        y_id: int = 1
        self.assertEqual(x.id, x_id)
        self.assertEqual(y.id, y_id)

        x_domain = Domain(0, MAX_ABSOLUTE_VALUE)
        y_domain = Domain(0, 1)
        self.assertEqual(x.global_domain, x_domain)
        self.assertEqual(y.global_domain, y_domain)

        x_var_type = VarType.INTEGER
        y_var_type = VarType.BINARY
        self.assertEqual(x.type, x_var_type)
        self.assertEqual(y.type, y_var_type)

    def test_add_constraint(self):
        model = Model()
        x: int = model.add_var()
        y: int = model.add_var(variable_type=VarType.BINARY)

        variable_indices: list[int] = [x, y]
        coefficients: list[float] = [1.0, -1.0]
        rhs: float = 1.0
        sense = Sense.EQ

        c: int = model.add_constraint(variable_indices,
                                      coefficients,
                                      rhs,
                                      sense)

        x: Variable = model.get_var(x)
        y: Variable = model.get_var(y)
        c: Constraint = model.get_constraint(c)

        self.assertEqual(1, x.column.size)
        self.assertEqual(0, x.column.get_constraint_index(0))
        self.assertEqual(1.0, x.column.get_coefficient(0))
        self.assertEqual(1, y.column.size)
        self.assertEqual(0, y.column.get_constraint_index(0))
        self.assertEqual(-1.0, y.column.get_coefficient(0))

        # test the constraint
        self.assertEqual(2, c.row.size)
        self.assertEqual(1.0, c.rhs)
        self.assertEqual(Sense.EQ, c.sense)
        self.assertListEqual(variable_indices, c.row._indices)
        self.assertListEqual(coefficients, c.row._coefficients)

    def test_apply_domain_changes(self):
        model = Model()
        x_id: int = model.add_var(upper_bound=10)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        c0_id: int = model.add_constraint([x_id, y_id],
                                          [1.0, 1.0],
                                          1.0)
        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)
        c0: Constraint = model.get_constraint(c0_id)

        self.assertEqual(0, c0.min_activity)
        self.assertEqual(11, c0.max_activity)

        previous_domain: Domain = x.local_domain
        new_domain = Domain(0, 5)

        # test applying the domain changes
        domain_change = DomainChange(x_id, previous_domain, new_domain)
        model.apply_domain_changes(domain_change)
        self.assertEqual(c0.max_activity, 6)
        self.assertEqual(x.local_domain, new_domain)

        model.apply_domain_changes(domain_change, undo=True)
        self.assertEqual(c0.max_activity, 11)
        self.assertEqual(x.local_domain, previous_domain)

    def test_convert_ge_inequalities(self):
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        c0_id: int = model.add_constraint([x_id, y_id],
                                          [1.0, 1.0],
                                          1.0,
                                          sense=Sense.GE)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)
        c0: Constraint = model.get_constraint(c0_id)
        row: Row = c0.row

        self.assertEqual(Sense.GE, c0.sense)
        self.assertEqual(1.0, c0.rhs)
        self.assertListEqual([x_id, y_id], row._indices)
        self.assertListEqual([1.0, 1.0], row._coefficients)

        self.assertEqual(1.0, x.column.get_coefficient(0))
        self.assertEqual(1.0, y.column.get_coefficient(0))

        # convert GE inequalities and check to see if it's correct
        model.convert_ge_constraints()
        self.assertEqual(Sense.LE, c0.sense)
        self.assertEqual(-1.0, c0.rhs)
        self.assertListEqual([x_id, y_id], row._indices)
        self.assertListEqual([-1.0, -1.0], row._coefficients)

        self.assertEqual(-1.0, x.column.get_coefficient(0))
        self.assertEqual(-1.0, y.column.get_coefficient(0))

    def test_init_model1(self):
        model = Model()
        x_id = model.add_var(variable_type=VarType.BINARY)
        y_id = model.add_var(variable_type=VarType.BINARY)

        c0_id = model.add_constraint([x_id, y_id],
                                     [1.0, 1.0],
                                     1.0,
                                     Sense.GE)
        model.init()

        c: Constraint = model.get_constraint(c0_id)
        self.assertEqual(0.0, c.min_activity)
        self.assertEqual(2.0, c.max_activity)

    def test_init_model2(self):
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)

        c0_id: int = model.add_constraint([x_id, y_id],
                                     [1.0, 1.0],
                                     1.0,
                                     Sense.GE)
        model.convert_ge_constraints()
        model.init()
        self.assertFalse(model.violated)

        c: Constraint = model.get_constraint(c0_id)
        self.assertEqual(-2.0, c.min_activity)
        self.assertEqual(0.0, c.max_activity)

        domain_changes: list[DomainChange] = [
            DomainChange.create_fixing(x, 0),
            DomainChange.create_fixing(y, 0)
        ]

        model.apply_domain_changes(*domain_changes)
        self.assertTrue(model.violated)

        model.apply_domain_changes(*domain_changes, undo=True)
        self.assertFalse(model.violated)
