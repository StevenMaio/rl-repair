from src.mip.model import *
from src.mip.propagation import LinearConstraintPropagator

from unittest import TestCase


class TestLinearConstraintPropagator(TestCase):

    def test_propagate1(self):
        """
        Tests to see if upper bounds are correctly modified based on the
        constraint activities for positive coefficients
        :return:
        """
        model = Model()
        propagator = LinearConstraintPropagator()

        x: int = model.add_var()
        y: int = model.add_var()
        c_id: int = model.add_constraint([x, y], [1.0, 2.0], 9)
        constraint: Constraint = model.get_constraint(c_id)
        self.assertEqual(0, constraint.min_activity)
        self.assertEqual(3e6, constraint.max_activity)
        self.assertFalse(constraint.propagated)

        domain_changes: list[DomainChange] = []
        propagator.propagate(model, constraint, domain_changes)

        x_new_domain = Domain(0, 9)
        y_new_domain = Domain(0, 4)

        self.assertTrue(constraint.propagated)
        self.assertEqual(x_new_domain, domain_changes[x].new_domain)
        self.assertEqual(y_new_domain, domain_changes[y].new_domain)

        model.apply_domain_changes(*domain_changes)
        self.assertFalse(constraint.propagated)
        self.assertEqual(0, constraint.min_activity)
        self.assertEqual(17, constraint.max_activity)

    def test_propagate2(self):
        """
        Tests to see if upper bounds are correctly modified based on the
        constraint activities for negative coefficients
        :return:
        """
        model = Model()
        propagator = LinearConstraintPropagator()

        x: int = model.add_var(lower_bound=0, upper_bound=10)
        y: int = model.add_var(lower_bound=0, upper_bound=5)
        c_id: int = model.add_constraint([x, y], [1.0, -2.0], -5)
        constraint: Constraint = model.get_constraint(c_id)

        self.assertEqual(-10, constraint.min_activity)
        self.assertEqual(10, constraint.max_activity)
        self.assertFalse(constraint.propagated)

        domain_changes: list[DomainChange] = []
        propagator.propagate(model, constraint, domain_changes)

        # expected new domains
        x_new_domain = Domain(0, 5)
        y_new_domain = Domain(3, 5)

        self.assertTrue(constraint.propagated)
        self.assertEqual(x_new_domain, domain_changes[x].new_domain)
        self.assertEqual(y_new_domain, domain_changes[y].new_domain)

        model.apply_domain_changes(*domain_changes)
        self.assertFalse(constraint.propagated)
        self.assertEqual(-10.0, constraint.min_activity)
        self.assertEqual(-1.0, constraint.max_activity)

    def test_propagate3(self):
        """
        Tests to see if propagation also works for equality constraints. This
        tests the constraint x + y == 3, -2 <= x <= 2 and 0 <= y <= 2.
        :return:
        """
        model = Model()
        propagator = LinearConstraintPropagator()

        x: int = model.add_var(lower_bound=-2, upper_bound=2)
        y: int = model.add_var(lower_bound=0, upper_bound=2)
        c_id: int = model.add_constraint([x, y], [1.0, 1.0], 3, Sense.EQ)
        constraint: Constraint = model.get_constraint(c_id)

        self.assertEqual(-2, constraint.min_activity)
        self.assertEqual(4, constraint.max_activity)
        self.assertFalse(constraint.propagated)

        domain_changes: list[DomainChange] = []
        propagator.propagate(model, constraint, domain_changes)

        x_new_domain = Domain(1, 2)
        y_new_domain = Domain(1, 2)

        self.assertTrue(constraint.propagated)
        self.assertEqual(x_new_domain, domain_changes[x].new_domain)
        self.assertEqual(y_new_domain, domain_changes[y].new_domain)

        model.apply_domain_changes(*domain_changes)
        self.assertFalse(constraint.propagated)
        self.assertEqual(2, constraint.min_activity)
        self.assertEqual(4, constraint.max_activity)

    def test_propagate4(self):
        """
        Tests to see if propagation also works for equality constraints. This
        tests the constraint x + y == 3, -2 <= x <= 2 and 0 <= y <= 2.
        :return:
        """
        model = Model()
        propagator = LinearConstraintPropagator()

        x: int = model.add_var(lower_bound=0, upper_bound=6)
        y: int = model.add_var(lower_bound=3, upper_bound=10)
        c_id: int = model.add_constraint([x, y], [1.0, -1.0], 2, Sense.EQ)
        constraint: Constraint = model.get_constraint(c_id)

        self.assertEqual(-10, constraint.min_activity)
        self.assertEqual(3, constraint.max_activity)
        self.assertFalse(constraint.propagated)

        domain_changes: list[DomainChange] = []
        propagator.propagate(model, constraint, domain_changes)

        x_new_domain = Domain(5, 6)
        y_new_domain = Domain(3, 4)

        self.assertTrue(constraint.propagated)
        self.assertEqual(x_new_domain, domain_changes[x].new_domain)
        self.assertEqual(y_new_domain, domain_changes[y].new_domain)

        model.apply_domain_changes(*domain_changes)
        self.assertFalse(constraint.propagated)
        self.assertEqual(1, constraint.min_activity)
        self.assertEqual(3, constraint.max_activity)
