"""
Handles linear constraint propagation.

TODO:
    - [ ] handle other senses for linear constraints (this isn't handled at
          the moment)
"""
import logging

from .Propagator import Propagator
from ..model import *

from typing import List

import math


class LinearConstraintPropagator(Propagator):
    _logger: logging.Logger

    def __init__(self):
        self._logger = logging.getLogger(__package__)

    def propagate(self,
                  model: Model,
                  constraint: Constraint,
                  propagation_changes: List[DomainChange]):
        """
        Deduces domain changes based on the min/max activity of the given
        constraint. The deduced domain changes are added to domain_changes.

        GE constraints are not handled. We assume that at this point, all GE
        inequalities have been transformed into LE inequalities.
        :param model:
        :param constraint:
        :param propagation_changes:
        """
        if constraint.propagated:
            return
        constraint.propagated = True
        rhs: float = constraint.rhs
        row: Row = constraint.row
        i: int
        for var_index, coefficient in row:
            var: Variable = model.get_var(var_index)
            residual_min_act: float
            residual_max_act: float
            if coefficient > 0:
                residual_min_act = constraint.min_activity - var.lb * coefficient
                lb: float = var.lb
                ub: float = (rhs - residual_min_act) / coefficient
                ub = max(lb, ub)
                new_domain = Domain(lb, ub)
                if var.type != VarType.CONTINUOUS:
                    ub = math.floor(ub)
                if constraint.sense == Sense.EQ:
                    residual_max_act = constraint.max_activity - coefficient * var.ub
                    lb = (rhs - residual_max_act) / coefficient
                    if var.type != VarType.CONTINUOUS:
                        lb = math.ceil(lb)
                if var.lb < lb or ub < var.ub:
                    lb = max(var.lb, lb)
                    ub = min(ub, var.ub)
                    prev_domain: Domain = var.local_domain
                    if new_domain not in prev_domain:
                        continue
                    domain_change = DomainChange(var.id, prev_domain, new_domain)
                    self._logger.debug('PROP_CHANGE cons_id=%d domain_change=%s coef>0',
                                       constraint.id,
                                       domain_change)
                    propagation_changes.append(domain_change)
            else:
                residual_min_act = constraint.min_activity - var.ub * coefficient
                lb: float = (rhs - residual_min_act) / coefficient
                ub: float = var.ub
                ub = max(lb, ub)
                new_domain = Domain(lb, ub)
                if var.type != VarType.CONTINUOUS:
                    lb = math.ceil(lb)
                if constraint.sense == Sense.EQ:
                    residual_max_act = constraint.max_activity - coefficient * var.lb
                    ub = (rhs - residual_max_act) / coefficient
                    if var.type != VarType.CONTINUOUS:
                        lb = math.ceil(lb)
                if var.lb < lb or ub < var.ub:
                    lb = max(var.lb, lb)
                    ub = min(ub, var.ub)
                    prev_domain: Domain = var.local_domain
                    if new_domain not in prev_domain:
                        continue
                    domain_change = DomainChange(var.id, prev_domain, new_domain)
                    self._logger.debug('PROP_CHANGE cons_id=%d domain_change=%s coef<0',
                                       constraint.id,
                                       domain_change)
                    propagation_changes.append(domain_change)
