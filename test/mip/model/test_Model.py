import logging
from unittest import TestCase

import gurobipy as gp
from gurobipy import GRB

from src.mip.model import *
from src.utils import initialize_logger

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
        m.addVar(vtype=GRB.BINARY, name="x3")
        x4 = m.addVar(vtype=GRB.CONTINUOUS, name="x4")
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

    def test_get_var(self):
        model: Model = self._instance
        counter: int = 0
        # test to see that the variables are sorted by type
        i: int
        for i in range(model.num_binary_variables):
            var: Variable = model.get_var(counter + i)
            self.assertEqual(var.variable_type, VarType.BINARY)
        counter = model.num_binary_variables
        for i in range(model.num_integer_variables):
            var: Variable = model.get_var(counter + i)
            self.assertEqual(var.variable_type, VarType.INTEGER)
        counter += model.num_integer_variables
        for i in range(model.num_continuous_variables):
            var: Variable = model.get_var(counter + i)
            self.assertEqual(var.variable_type, VarType.CONTINUOUS)

        # test column of variable 1
        x1: Variable = model.get_var(1)
        column: Column = x1.column
        actual_size: int = 3
        actual_indices: list[int] = [0, 1, 2]
        actual_coefficients: list[float] = [3.0, -3.0, 1.0]
        self.assertEqual(column.size, actual_size)
        self.assertListEqual(column._indices, actual_indices)
        self.assertListEqual(column._coefficients, actual_coefficients)

    def test_get_constraint(self):
        model: Model = self._instance

        # expected values after variable sorting for constraint 0
        c: Constraint = model.get_constraint(0)
        row: Row = c.row
        actual_size: int = 2
        actual_rhs: float = 6.0
        actual_indices: list[int] = [1, 2]
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
        actual_indices: list[int] = [1, 2]
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
        actual_indices: list[int] = [1]
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
        actual_indices: list[int] = [3]
        actual_coefficients: list[float] = [1]
        self.assertEqual(row.size, actual_size)
        self.assertEqual(c.rhs, actual_rhs)
        self.assertEqual(c.sense, Sense.EQ)
        self.assertListEqual(row._indices, actual_indices)
        self.assertListEqual(row._coefficients, actual_coefficients)
