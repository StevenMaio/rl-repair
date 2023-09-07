from typing import Union


class Domain:
    EMPTY_DOMAIN: "Domain"
    _lower_bound: float
    _upper_bound: float

    def __init__(self, lower_bound: float, upper_bound: float):
        self._lower_bound = float(lower_bound)
        self._upper_bound = float(upper_bound)

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    def copy(self) -> "Domain":
        copy = Domain(self._lower_bound, self._upper_bound)
        return copy

    def __eq__(self, other: "Domain"):
        if isinstance(other, Domain):
            return self._lower_bound == other._lower_bound and self._upper_bound == other._upper_bound
        else:
            return False

    def __repr__(self) -> str:
        return f'[{self._lower_bound}, {self._upper_bound}]'

    def intersects(self, other: "Domain") -> bool:
        return not (other.upper_bound < self.lower_bound or other.lower_bound > self.upper_bound)

    def compute_intersection(self, other: "Domain") -> "Domain":
        if not self.intersects(other):
            return Domain.EMPTY_DOMAIN
        else:
            lb: float = max(self.lower_bound, other.lower_bound)
            ub: float = max(min(self.upper_bound, other.upper_bound),
                            lb)
            return Domain(lb, ub)

    def __add__(self, shift: float) -> "Domain":
        new_domain = Domain(self.lower_bound + shift,
                            self.upper_bound + shift)
        return new_domain

    @staticmethod
    def singleton(value: float) -> "Domain":
        singleton = Domain(value, value)
        return singleton

    def size(self) -> float:
        return self._upper_bound - self._lower_bound

    def __contains__(self, item: Union["Domain", float]):
        if isinstance(item, float):
            return self._lower_bound <= item <= self._upper_bound
        else:
            return self._lower_bound <= item._lower_bound and item._upper_bound <= self._upper_bound


Domain.EMPTY_DOMAIN = -1


class DomainChange:
    _var_id: int
    _previous_domain: Domain
    _new_domain: Domain

    def __init__(self, var_id: int, previous_domain: Domain, new_domain: Domain):
        self._var_id = var_id
        self._previous_domain = previous_domain
        self._new_domain = new_domain

    @property
    def var_id(self) -> int:
        return self._var_id

    @property
    def previous_domain(self) -> Domain:
        return self._previous_domain

    @property
    def new_domain(self) -> Domain:
        return self._new_domain

    def __repr__(self) -> str:
        return f'DomainChange(var_id={self.var_id}, prev_domain={self.previous_domain} new_domain={self.new_domain})'

    @staticmethod
    def create_fixing(var: "Variable", value: float) -> "DomainChange":
        fixed_domain = Domain(value, value)
        fixing = DomainChange(var.id, var.local_domain, fixed_domain)
        return fixing
