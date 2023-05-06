"""
Handles linear constraint propagation.

TODO:
    - [ ] handle other senses for linear constraints (this isn't handled at
          the moment)
"""
from .Propagator import Propagator
from ..model import *

import math


class LinearConstraintPropagator(Propagator):

    def propagate(self,
                  model: Model,
                  constraint: Constraint,
                  domain_changes: list[DomainChange]):
        """
        Deduces domain changes based on the min/max activity of the given
        constraint. The deduced domain changes are added to domain_changes.

        GE constraints are not handled. We assume that at this point, all GE
        inequalities have been transformed into LE inequalities.
        :param model:
        :param constraint:
        :param domain_changes:
        """
        if constraint.propagated:
            return
        constraint.propagated = True
        rhs: float = constraint.rhs
        row: Row = constraint.row
        i: int
        for i in range(row.size):
            var_index: int = row.get_var_index(i)
            var: Variable = model.get_var(var_index)
            coefficient: float = row.get_coefficient(i)
            residual_min_act: float
            residual_max_act: float
            if coefficient > 0:
                residual_min_act = constraint.min_activity - var.lb * coefficient
                lb: float = var.lb
                ub: float = (rhs - residual_min_act) / coefficient
                if var.variable_type != VarType.CONTINUOUS:
                    ub = math.floor(ub)
                if constraint.sense == Sense.EQ:
                    residual_max_act = constraint.max_activity - coefficient * var.ub
                    lb = (rhs - residual_max_act) / coefficient
                    if var.variable_type != VarType.CONTINUOUS:
                        lb = math.ceil(lb)
                if var.lb < lb or ub < var.ub:
                    lb = max(var.lb, lb)
                    ub = min(ub, var.ub)
                    prev_domain: Domain = var.local_domain
                    new_domain = Domain(lb, ub)
                    domain_change = DomainChange(var.id, prev_domain, new_domain)
                    domain_changes.append(domain_change)
            else:
                residual_min_act = constraint.min_activity - var.ub * coefficient
                lb: float = (rhs - residual_min_act) / coefficient
                ub: float = var.ub
                if var.variable_type != VarType.CONTINUOUS:
                    lb = math.ceil(lb)
                if constraint.sense == Sense.EQ:
                    residual_max_act = constraint.max_activity - coefficient * var.lb
                    ub = (rhs - residual_max_act) / coefficient
                    if var.variable_type != VarType.CONTINUOUS:
                        lb = math.ceil(lb)
                if var.lb < lb or ub < var.ub:
                    lb = max(var.lb, lb)
                    ub = min(ub, var.ub)
                    prev_domain: Domain = var.local_domain
                    new_domain = Domain(lb, ub)
                    domain_change = DomainChange(var.id, prev_domain, new_domain)
                    domain_changes.append(domain_change)
