"""
Contains the Node class implementation for this project.
"""
from enum import Enum, auto
from typing import List

import torch

from src.mip.model import VarType, Sense
from src.rl.params import GnnParams
from src.utils import IndexEnum


class FeatIdx(IndexEnum):
    """
    Class that maps features to their respective indices.
    """
    # variable features
    IS_BINARY = auto()
    IS_INTEGER = auto()
    IS_CONTINUOUS = auto()
    LOCAL_DOMAIN_SIZE = auto()
    IS_FIXED = auto()

    # constraint features
    IS_LE_CONSTRAINT = auto()
    IS_EQ_CONSTRAINT = auto()
    VIOLATION = auto()
    NUM_VARIABLES = auto()


class NodeType(Enum):
    VAR = auto()
    CONS = auto()


class Node:
    _edges: List["Edge"]
    _data: object
    _features: torch.Tensor
    _node_type: NodeType

    def __init__(self, data: object, node_type: NodeType):
        self._edges = []
        self._data = data
        self._node_type = node_type
        if node_type == NodeType.VAR or node_type == NodeType.CONS:
            self.update(None, True)
        else:
            raise Exception("Unsupported NodeType value")

    def update(self, model, initialize=False):
        if self.type == NodeType.VAR:
            self._update_var_node(model, initialize)
        else:
            self._update_cons_node(model, initialize)

    def _update_var_node(self, model, initialize):
        var = self._data
        if initialize:
            self._features = torch.zeros(GnnParams.num_node_features)
            if var.type == VarType.BINARY:
                self._features[FeatIdx.IS_BINARY] = 1.0
            elif var.type == VarType.INTEGER:
                self._features[FeatIdx.IS_INTEGER] = 1.0
            else:
                self._features[FeatIdx.IS_CONTINUOUS] = 1.0
        if model is None:
            return
        self._features[FeatIdx.LOCAL_DOMAIN_SIZE] = var.local_domain.size() / var.global_domain.size()
        self._features[FeatIdx.IS_FIXED] = 1 if var.lb == var.ub else 0

    def _update_cons_node(self, model, initialize):
        cons = self._data
        if initialize:
            self._features = torch.zeros(GnnParams.num_node_features)
            if cons.sense == Sense.LE:
                self._features[FeatIdx.IS_LE_CONSTRAINT] = 1.0
            elif cons.sense == Sense.EQ:
                self._features[FeatIdx.IS_EQ_CONSTRAINT] = 1.0
        if model is None:
            return
        self._features[FeatIdx.NUM_VARIABLES] = cons.row.size / model.largest_cons_size

    @property
    def edges(self) -> List["Edge"]:
        return self._edges

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def type(self) -> NodeType:
        return self._node_type

    @property
    def data_id(self):
        return self._data.id
