from src.mip.model import Model

from itertools import chain
from typing import List, Tuple, Dict

from src.rl.graph.Node import Node, NodeType
from src.rl.graph.Edge import Edge


class Graph:
    _var_nodes: List[Node]
    _cons_nodes: List[Node]
    _edges: Dict[Tuple[int, int], Edge]  # always indexed by (var_id, cons_id)

    def __init__(self, model: Model):
        self._var_nodes = []
        self._cons_nodes = []
        self._edges = {}
        for var in model.variables:
            self._var_nodes.append(Node(model, var, NodeType.VAR))
        for cons in model.constraints:
            cons_node = Node(model, cons, NodeType.CONS)
            self._cons_nodes.append(cons_node)
            row = cons.row
            for var_id, coef in row:
                var_node = self._var_nodes[var_id]
                e = Edge(var_node, cons_node, coef)
                cons_node.edges.append(e)
                var_node.edges.append(e)
                self._edges[(var_id, cons.id)] = e

    @property
    def var_nodes(self) -> List[Node]:
        return self._var_nodes

    @property
    def cons_nodes(self) -> List[Node]:
        return self._cons_nodes

    @property
    def edges(self) -> Dict[Tuple[int, int], Edge]:
        return self._edges

    def update(self, model):
        for u in chain(self._var_nodes, self._cons_nodes):
            u.update(model)
