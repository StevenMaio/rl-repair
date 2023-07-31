DEFAULT_NOISE_PARAMETER: float = 0.75
DEFAULT_MAX_ITERATIONS: int = 200
DEFAULT_SOFT_RESET_LIMIT: int = 200
DEFAULT_MAX_HISTORY: int = 3


class RepairWalkParams:
    _noise_parameter: float
    _max_iterations: int
    _soft_reset_limit: int
    _max_history: int

    def __init__(self,
                 noise_parameter: float = DEFAULT_NOISE_PARAMETER,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 soft_reset_limit: int = DEFAULT_SOFT_RESET_LIMIT,
                 max_history: int = DEFAULT_MAX_HISTORY):
        self._noise_parameter = noise_parameter
        self._max_iterations = max_iterations
        self._soft_reset_limit = soft_reset_limit
        self._max_history = max_history

    @property
    def noise_parameter(self) -> float:
        return self._noise_parameter

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def soft_reset_limit(self) -> int:
        return self._soft_reset_limit

    @property
    def history_size(self) -> int:
        return self._max_history
