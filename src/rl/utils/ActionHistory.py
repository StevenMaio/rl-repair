from enum import Enum, auto

from typing import List, Tuple


class ActionType(Enum):
    FIXING = auto()
    REPAIR = auto()


class ActionHistory:
    """
    Represents the move made by FPRL. If the action is a fixing, then the info is a tuple
    (var_id, b) where var_id is the id of the var being fixed, and b is a binary variable
    such that b=0 of the variables was fixed first to its lower bound, and b=1 if it was
    fixed first to its upper bound.

    For a repair move, the action is a tuple (cons_id, var_id) where cons_id is the id of
    the constraint being repaired, and var_id is the id of the variable being shifted.
    """
    _in_training: bool
    _moves: List[Tuple[object, ActionType]]

    def __init__(self, in_training: bool):
        self._in_training = in_training
        self._moves = []

    def add(self, action, action_type):
        if self._in_training:
            self._moves.append((action, action_type))

    def clear(self):
        self._moves.clear()

    @property
    def moves(self):
        return self._moves

    def __iter__(self):
        return iter(self._moves)
