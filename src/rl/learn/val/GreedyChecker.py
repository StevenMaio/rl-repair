from .ValidationProgressChecker import ValidationProgressChecker


class GreedyChecker(ValidationProgressChecker):
    _best_val_score: float
    _num_worse_iters: int
    _max_num_worse_iters: int

    def __init__(self, max_num_worse_iters):
        self._max_num_worse_iters = max_num_worse_iters

    def update_progress(self, val_score: float):
        if val_score > self._best_val_score:
            self._best_val_score = val_score
            self._num_worse_iters = 0
        else:
            self._num_worse_iters += 1

    def continue_training(self) -> bool:
        return self._num_worse_iters < self._max_num_worse_iters

    def reset(self):
        self._best_val_score = 0

    def corrected_score(self):
        return self._best_val_score
