from .Propagator import Propagator
from ..model import *

import math


class LinearConstraintPropagator(Propagator):
    def propagate(self,
                  model: Model,
                  constraint: Constraint,
                  domain_changes: list[DomainChange]):
        if constraint.propagated:
            return
        constraint.propagated = True
        row: Row = constraint.row
        i: int
        for i in range(row.size):
            var_index: int = row.get_var_index(i)
            var: Variable = model.get_var(var_index)
            coefficient: float = row.get_coefficient(i)
            residual_min_act: float
            if coefficient > 0:
                residual_min_act = constraint.min_activity - var.lb * coefficient
                ub: float = (constraint.rhs - residual_min_act) / coefficient
                if var.variable_type != VarType.CONTINUOUS:
                    ub = math.floor(ub)
                if ub < var.ub:
                    prev_domain: Domain = var.local_domain.copy()
                    new_domain = Domain(var.lb, ub)
                    domain_change = DomainChange(var.id, prev_domain, new_domain)
                    domain_changes.append(domain_change)
            else:
                residual_min_act = constraint.min_activity - var.ub * coefficient
                lb: float = (constraint.rhs - residual_min_act) / coefficient
                if var.variable_type != VarType.CONTINUOUS:
                    lb = math.ceil(lb)
                if lb > var.lb:
                    prev_domain: Domain = var.local_domain.copy()
                    new_domain = Domain(lb, var.ub)
                    domain_change = DomainChange(var.id, prev_domain, new_domain)
                    domain_changes.append(domain_change)
