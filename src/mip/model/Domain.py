class Domain:
    _lower_bound: float
    _upper_bound: float

    def __init__(self, lower_bound: float, upper_bound: float):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

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
        return self._lower_bound == other._lower_bound and self._upper_bound == other._upper_bound


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
