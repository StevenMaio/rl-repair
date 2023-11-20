"""
Contains the Node class implementation for this project.
"""
from enum import Enum, auto
from typing import List, Union

import torch

from src.mip.model import VarType, Sense, Variable, Constraint, Model
from src.rl.params import GnnParams
from src.utils import IndexEnum


class VarFeatIdx(IndexEnum):
    """
    Class that maps variable features to their respective indices.
    """
    IS_BINARY = auto()
    IS_INTEGER = auto()
    IS_CONTINUOUS = auto()
    LOCAL_DOMAIN_SIZE = auto()
    LOCAL_LOWER_BOUND = auto()
    LOCAL_UPPER_BOUND = auto()
    IS_FIXED = auto()
    APPEARANCE_SCORE = auto()
    OBJECTIVE_COEF = auto()
    PORTION_UP_LOCKS = auto()
    PORTION_DOWN_LOCKS = auto()


class ConsFeatIdx(IndexEnum):
    """
    Class that maps constraint features to their respective indices.
    """
    IS_LE_CONSTRAINT = auto()
    IS_EQ_CONSTRAINT = auto()
    VIOLATION = auto()
    NUM_VARIABLES = auto()
    IS_FEASIBLE = auto()
    IS_VIOLATED = auto()
    PORTION_FIXED_VARIABLES = auto()
    MIN_ACTIVITY = auto()
    MAX_ACTIVITY = auto()


class NodeType(Enum):
    VAR = auto()
    CONS = auto()


class Node:
    _edges: List["Edge"]
    _data: Union[Variable, Constraint]
    _features: torch.Tensor
    _node_type: NodeType

    def __init__(self,
                 model: Model,
                 data: Union[Variable, Constraint],
                 node_type: NodeType):
        self._edges = []
        self._data = data
        self._node_type = node_type
        if node_type == NodeType.VAR or node_type == NodeType.CONS:
            self.update(model, True)
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
            self._features = torch.zeros(GnnParams.num_var_node_features)
            if var.type == VarType.BINARY:
                self._features[VarFeatIdx.IS_BINARY] = 1.0
            elif var.type == VarType.INTEGER:
                self._features[VarFeatIdx.IS_INTEGER] = 1.0
            else:
                self._features[VarFeatIdx.IS_CONTINUOUS] = 1.0
            self._features[VarFeatIdx.OBJECTIVE_COEF] = var.objective_coefficient
            self._features[VarFeatIdx.APPEARANCE_SCORE] = var.column.size / len(model.constraints)
            if var.column.size == 0:
                self._features[VarFeatIdx.PORTION_UP_LOCKS] = 0
                self._features[VarFeatIdx.PORTION_DOWN_LOCKS] = 0
            else:
                self._features[VarFeatIdx.PORTION_UP_LOCKS] = var.num_up_locks / var.column.size
                self._features[VarFeatIdx.PORTION_DOWN_LOCKS] = var.num_down_locks / var.column.size
        self._features[VarFeatIdx.LOCAL_DOMAIN_SIZE] = var.local_domain.size() / var.global_domain.size()
        self._features[VarFeatIdx.LOCAL_UPPER_BOUND] = var.ub
        self._features[VarFeatIdx.LOCAL_LOWER_BOUND] = var.lb
        self._features[VarFeatIdx.IS_FIXED] = 1 if var.lb == var.ub else 0

    def _update_cons_node(self, model, initialize):
        cons = self._data
        if initialize:
            self._features = torch.zeros(GnnParams.num_cons_node_features)
            if cons.sense == Sense.LE:
                self._features[ConsFeatIdx.IS_LE_CONSTRAINT] = 1.0
            elif cons.sense == Sense.EQ:
                self._features[ConsFeatIdx.IS_EQ_CONSTRAINT] = 1.0
            self._features[ConsFeatIdx.NUM_VARIABLES] = cons.row.size / model.largest_cons_size
        self._features[ConsFeatIdx.IS_VIOLATED] = cons.is_violated()
        self._features[ConsFeatIdx.IS_FEASIBLE] = 1 - cons.is_violated()
        self._features[ConsFeatIdx.VIOLATION] = cons.compute_violation()
        self._features[ConsFeatIdx.MIN_ACTIVITY] = cons.min_activity
        self._features[ConsFeatIdx.MAX_ACTIVITY] = cons.max_activity

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
