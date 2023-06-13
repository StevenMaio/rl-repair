from typing import Callable, Tuple, List

from .RepairStrategy import RepairStrategy

import random
import logging
import math

from ...model import Sense, VarType, Domain, DomainChange

from src.utils import compute_interval_distance, REPAIR_LEVEL
from src.utils.data_struct.CircularList import CircularList


def num_violated_constraints(model: "Model") -> float:
    """
    Violation score that is equal to the number of violated constraints
    :param model:
    :return:
    """
    return len(list(filter(lambda c: c.is_violated(), model.constraints)))


class RepairWalk(RepairStrategy):
    name: str = 'RepairWalk'

    # configuration
    _max_iterations: int
    _soft_reset_limit: int
    _noise_parameter: float
    _history_size: int

    _violation_scorer: Callable[["Model"], float]
    _logger: logging.Logger

    def __init__(self,
                 params: "RepairWalkParams",
                 **kwargs):
        self._max_iterations = params.max_iterations
        self._soft_reset_limit = params.soft_reset_limit
        self._noise_parameter = params.noise_parameter
        self._history_size = params.history_size
        self._violation_scorer = num_violated_constraints
        self._logger = logging.getLogger(__package__)
        self._logger.setLevel(REPAIR_LEVEL)
        self._logger.log(REPAIR_LEVEL,
                         'initialized max_iterations=%d soft_reset_limit=%d noise_parameter=%.2f max_history=%d',
                         params.max_iterations,
                         params.soft_reset_limit,
                         params.noise_parameter,
                         params.history_size)

    def repair_domain(self,
                      model: "Model",
                      repair_changes: List["DomainChange"]):
        best_violation_score: float = self._violation_scorer(model)
        best_repair_changes: List["DomainChange"] = []
        reset_changes: List["DomainChange"] = []
        soft_reset_counter: int = 0
        success: bool = False
        shift_history = CircularList(self._history_size)
        for iter_num in range(self._max_iterations):
            model.update()
            cons = self._sample_violated_constraint(model)
            var, domain_change = self._select_shift_candidate(model, cons)
            if var is None or domain_change in shift_history:
                continue
            self._logger.log(REPAIR_LEVEL,
                             'iter_num=%d cons=%s shift=%s',
                             iter_num,
                             cons,
                             domain_change)
            shift_history.add(domain_change)
            model.apply_domain_changes(domain_change)
            reset_changes.append(domain_change)
            violation_score: float = self._violation_scorer(model)
            if violation_score == 0:
                best_repair_changes.extend(reset_changes)
                success = True
                break
            elif violation_score < best_violation_score:
                soft_reset_counter = 0
                best_violation_score = violation_score
                best_repair_changes.extend(reset_changes)
                reset_changes.clear()
            else:
                soft_reset_counter += 1
                if soft_reset_counter == self._soft_reset_limit:
                    self._logger.log(REPAIR_LEVEL,
                                     'soft reset limit hit best_score=%.2f',
                                     best_violation_score)
                    soft_reset_counter = 0
                    model.apply_domain_changes(*reset_changes, undo=True)
                    reset_changes.clear()
        if success:
            repair_changes.extend(best_repair_changes)
            model.violated = False
        else:
            if len(best_repair_changes) > 0:
                model.apply_domain_changes(*best_repair_changes, undo=True)
        return success

    def _sample_violated_constraint(self, model: "Model") -> "Constraint":
        violated_constraints = list(filter(lambda c: c.is_violated(), model.constraints))
        return random.choice(violated_constraints)

    def _select_shift_candidate(self,
                                model: "Model",
                                constraint: "Constraint") -> Tuple["Variable", "DomainChange"]:
        shift_candidates = []
        has_plateau_move: bool = False
        row: "Row" = constraint.row
        for var_id, coefficient in row:
            var: "Variable" = model.get_var(var_id)
            should_shift: bool
            shift_amount: float
            if var.type == VarType.BINARY:
                should_shift, shift_amount = self._determine_binary_var_shift_amount(constraint,
                                                                                           var,
                                                                                           coefficient)
            elif var.type == VarType.INTEGER:
                should_shift, shift_amount = self._determine_integer_var_shift_amount(constraint,
                                                                                            var,
                                                                                            coefficient)
            else:
                continue
            if should_shift:
                shifted_domain = var.global_domain.compute_intersection(var.local_domain + shift_amount)
                shift_damage = self._compute_shift_damage(var, shifted_domain, model)
                has_plateau_move |= (shift_damage == 0)
                shift_candidates.append((var, shifted_domain, shift_damage))
        if has_plateau_move:
            plateau_moves = list(filter(lambda t: t[2] == 0, shift_candidates))
            var, new_domain = self._sample_var_candidate(model, constraint, plateau_moves)
            domain_change = DomainChange(var.id, var.local_domain, new_domain)
            return var, domain_change
        elif len(shift_candidates) > 0:
            if random.random() <= self._noise_parameter:
                var, new_domain = self._sample_var_candidate(model, constraint, shift_candidates)
            else:
                var, new_domain, _ = min(shift_candidates, key=lambda t: t[2])
            domain_change = DomainChange(var.id, var.local_domain, new_domain)
            return var, domain_change
        else:
            return None, None

    @staticmethod
    def _compute_shift_damage(var: "Variable", shifted_domain: "Domain", model: "Model") -> float:
        lb_shift = shifted_domain.lower_bound - var.lb
        ub_shift = shifted_domain.upper_bound - var.ub
        shift_damage: float = 0
        for con_id, coef in var.column:
            cons: "Constraint" = model.get_constraint(con_id)
            if coef > 0:
                min_activity = cons.min_activity + lb_shift * coef
                max_activity = cons.max_activity + ub_shift * coef
            else:
                min_activity = cons.min_activity + ub_shift * coef
                max_activity = cons.max_activity + lb_shift * coef
            shift_damage += int(cons.is_violated(min_activity, max_activity))
        return shift_damage

    @staticmethod
    def _determine_integer_var_shift_amount(cons: "Constraint",
                                            var: "Variable",
                                            coef: float):
        global_domain: "Domain" = var.global_domain
        local_domain: "Domain" = var.local_domain
        if global_domain == local_domain:
            return False, 0
        if cons.sense == Sense.LE:
            shift: float = (cons.rhs - cons.min_activity) / coef
            if coef < 0:
                shift = math.floor(shift)
            else:
                shift = math.ceil(shift)
            shifted_domain = local_domain + shift
            intersection = global_domain.compute_intersection(shifted_domain)
            if intersection != Domain.EMPTY_DOMAIN:
                lb_shift: float = intersection.lower_bound - local_domain.lower_bound
                ub_shift: float = intersection.upper_bound - local_domain.upper_bound
                if coef > 0:
                    min_activity: float = cons.min_activity + lb_shift * coef
                else:
                    min_activity: float = cons.min_activity + ub_shift * coef
                return (min_activity <= cons.rhs), shift
            else:
                return False, 0
        elif cons.sense == Sense.EQ:
            original_distance: float = compute_interval_distance(cons.min_activity,
                                                                 cons.max_activity,
                                                                 cons.rhs)
            shift: float
            if cons.rhs < cons.min_activity:
                shift = (cons.rhs - cons.min_activity) / coef
                if coef < 0:
                    shift = math.floor(shift)
                else:
                    shift = math.ceil(shift)
            else:
                shift = (cons.rhs - cons.max_activity) / coef
                if coef < 0:
                    shift = math.ceil(shift)
                else:
                    shift = math.floor(shift)
            shifted_domain = local_domain + shift
            intersection = global_domain.compute_intersection(shifted_domain)
            if intersection != Domain.EMPTY_DOMAIN:
                lb_shift: float = intersection.lower_bound - local_domain.lower_bound
                ub_shift: float = intersection.upper_bound - local_domain.upper_bound
                if coef > 0:
                    min_activity: float = cons.min_activity + lb_shift * coef
                    max_activity: float = cons.max_activity + ub_shift * coef
                else:
                    min_activity: float = cons.min_activity + ub_shift * coef
                    max_activity: float = cons.max_activity + lb_shift * coef
                new_distance: float = compute_interval_distance(min_activity,
                                                                max_activity,
                                                                cons.rhs)
                return (new_distance < original_distance), shift
            else:
                return False, 0
        else:
            raise Exception("Sense.GE not supported")

    def _sample_var_candidate(self, model, constraint, candidates):
        """

        :param candidates:
        :return:
        """
        var, new_domain, _ = random.choice(candidates)
        return var, new_domain

    @staticmethod
    def _determine_binary_var_shift_amount(cons: "Constraint",
                                           var: "Variable",
                                           coef: float):
        """
        Helper method that returns a tuple (should_shift, shift_amount) such
        that should_shift indicates whether the variable can be shifted, and
        shift_amount is the amount by which the variable should be shifted
        :param cons:
        :param var:
        :param coef:
        :return:
        """
        lb: float = var.lb
        ub: float = var.ub
        shift: float = 0
        if lb != ub:
            return False, shift
        # determine the shift amount
        if lb == 0:
            shift = 1
        else:
            shift = -1
        if cons.sense == Sense.LE:
            if coef > 0 and shift == -1:
                return True, shift
            elif coef < 0 and shift == 1:
                return True, shift
            else:
                return False, shift
        elif cons.sense == Sense.EQ:
            original_distance: float = compute_interval_distance(cons.min_activity,
                                                                 cons.max_activity,
                                                                 cons.rhs)
            min_activity: float = cons.min_activity + shift * coef
            max_activity: float = cons.max_activity + shift * coef
            new_distance: float = compute_interval_distance(min_activity,
                                                            max_activity,
                                                            cons.rhs)
            return (new_distance < original_distance), shift
        else:
            raise Exception("Sense.GE not supported")
