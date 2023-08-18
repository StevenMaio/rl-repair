import torch
from enum import auto

from src.rl.params.GnnParams import DEFAULT_NUM_EDGE_FEATURES
from src.utils import IndexEnum


class EdgeFeatIdx(IndexEnum):
    COEF_MAGNITUDE = auto()
    COEF_IS_POSITIVE = auto()
    COEF_IS_NEGATIVE = auto()


class Edge:
    _var_node: "Node"
    _cons_node: "Node"
    _features: torch.Tensor

    def __init__(self,
                 var_node: "Node",
                 cons_node: "Node",
                 coef: float) -> object:
        self._var_node = var_node
        self._cons_node = cons_node
        self._features = torch.zeros(DEFAULT_NUM_EDGE_FEATURES)
        self._features[EdgeFeatIdx.COEF_MAGNITUDE] = abs(coef)
        if coef > 0:
            self._features[EdgeFeatIdx.COEF_IS_POSITIVE] = 1.0
        else:
            self._features[EdgeFeatIdx.COEF_IS_NEGATIVE] = 1.0

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def var_node(self) -> "Node":
        return self._var_node

    @property
    def cons_node(self) -> "Node":
        return self._cons_node
