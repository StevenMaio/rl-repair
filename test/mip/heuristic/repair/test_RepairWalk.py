""""
Unittests for RepairWalk.

TODO: Need to add tests for the following things
    1. continuous variables aren't touched (yes?)
    2. LE constraints
        a. positive shifts
        b. negative shifts
    3. EQ constraints
        - everything
    4. plateau moves have priority (yes?)
    5. variable shifts stay within bounds.
"""
from unittest import TestCase

from src.mip.model import Model, VarType, Sense, DomainChange, Domain, Variable
from src.mip.heuristic.repair.RepairWalk import RepairWalk
from src.mip.params import RepairWalkParams

import random
import logging

from src.utils import initialize_logger, REPAIR_LEVEL


initialize_logger(level=REPAIR_LEVEL)


class TestRepairWalk(TestCase):

    @classmethod
    def setUpClass(cls):
        cls._logger = logging.getLogger(__package__)

    def test_simple_shift(self):
        """
        This test tests the model x_1 + x_2 == 1 with x_1 = x_2 = 0 (x_1, x_2
        are binary variables). One of the variables should be shifted to 1, and
        the list of repair changes should contain one element.
        """
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)

        model.init()
        self.assertFalse(model.violated)

        domain_changes = [
            DomainChange(x_id, x.local_domain, Domain.singleton(0)),
            DomainChange(y_id, y.local_domain, Domain.singleton(0))
        ]
        model.apply_domain_changes(*domain_changes)
        self.assertTrue(model.violated)

        params = RepairWalkParams()
        repair_walk = RepairWalk(params)
        repair_changes = []
        success: bool = repair_walk.repair_domain(model, repair_changes)
        self.assertTrue(success)
        self.assertEqual(1, len(repair_changes))

    def test_plateau_move_priority(self):
        """
        This test checks to see if plateau moves are given priority.
        """
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        z_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)
        z: Variable = model.get_var(z_id)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([y_id, z_id],
                             [1.0, -1.0],
                             0.0,
                             Sense.EQ)

        model.init()
        self.assertFalse(model.violated)

        domain_changes = [
            DomainChange(x_id, x.local_domain, Domain.singleton(0)),
            DomainChange(y_id, y.local_domain, Domain.singleton(0)),
            DomainChange(z_id, z.local_domain, Domain.singleton(0))
        ]
        model.apply_domain_changes(*domain_changes)
        self.assertTrue(model.violated)

        params = RepairWalkParams()
        repair_walk = RepairWalk(params)
        repair_changes = []
        success: bool = repair_walk.repair_domain(model, repair_changes)
        self.assertTrue(success)
        self.assertEqual(1, x.lb)
        self.assertEqual(1, len(repair_changes))

    def test_ignore_continuous_variables(self):
        """
        Checks to see that continuous variables are ignored.
        """
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.CONTINUOUS,
                                  lower_bound=0,
                                  upper_bound=1)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        z_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)
        z: Variable = model.get_var(z_id)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([y_id, z_id],
                             [1.0, -1.0],
                             0.0,
                             Sense.EQ)

        model.init()
        self.assertFalse(model.violated)

        domain_changes = [
            DomainChange(x_id, x.local_domain, Domain.singleton(0)),
            DomainChange(y_id, y.local_domain, Domain.singleton(0)),
            DomainChange(z_id, z.local_domain, Domain.singleton(0))
        ]
        model.apply_domain_changes(*domain_changes)
        self.assertTrue(model.violated)

        params = RepairWalkParams()
        repair_walk = RepairWalk(params)
        repair_changes = []
        success: bool = repair_walk.repair_domain(model, repair_changes)

        # the repair should be successful, and two repair moves should be made
        self.assertTrue(success)
        self.assertEqual(1, y.lb)
        self.assertEqual(1, z.lb)
        self.assertEqual(2, len(repair_changes))

    def test_soft_reset(self):
        """
        Checks to see if the soft reset forces RepairWalk to undo the current changes.
        """
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.CONTINUOUS,
                                  lower_bound=0,
                                  upper_bound=1)
        y_id: int = model.add_var(variable_type=VarType.BINARY)
        z_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)
        z: Variable = model.get_var(z_id)

        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([y_id, z_id],
                             [1.0, -1.0],
                             0.0,
                             Sense.EQ)

        model.init()
        self.assertFalse(model.violated)

        domain_changes = [
            DomainChange(x_id, x.local_domain, Domain.singleton(0)),
            DomainChange(y_id, y.local_domain, Domain.singleton(0)),
            DomainChange(z_id, z.local_domain, Domain.singleton(0))
        ]
        model.apply_domain_changes(*domain_changes)
        self.assertTrue(model.violated)

        params = RepairWalkParams(soft_reset_limit=1)
        repair_walk = RepairWalk(params)
        repair_changes = []
        success: bool = repair_walk.repair_domain(model, repair_changes)
        self.assertFalse(success)

    def test_negative_shift(self):
        """
        Checks to see if the repair does negative shifts
        """
        rng_seed: int = 0
        self._logger.info('starting test rng_seed=%d', rng_seed)
        random.seed(rng_seed)
        model = Model()

        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.BINARY)

        x: Variable = model.get_var(x_id)
        y: Variable = model.get_var(y_id)

        model.add_constraint([x_id, y_id],
                             [1.0, -1.0],
                             -1.0,
                             Sense.EQ)

        model.init()
        self.assertFalse(model.violated)

        domain_changes = [
            DomainChange(x_id, x.local_domain, Domain.singleton(1)),
            DomainChange(y_id, y.local_domain, Domain.singleton(1)),
        ]
        model.apply_domain_changes(*domain_changes)
        self.assertTrue(model.violated)

        params = RepairWalkParams()
        repair_walk = RepairWalk(params)
        repair_changes = []
        success: bool = repair_walk.repair_domain(model, repair_changes)
        self.assertTrue(success)
        self.assertEqual(0, x.ub)
        self.assertEqual(1, len(repair_changes))
        self.assertFalse(model.violated)
