import torch


class Edge:
    _var_node: "Node"
    _cons_node: "Node"
    _features: torch.Tensor

    def __init__(self,
                 var_node: "Node",
                 cons_node: "Node",
                 coef: float):
        self._var_node = var_node
        self._cons_node = cons_node
        self._features = torch.Tensor([coef])

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def var_node(self) -> "Node":
        return self._var_node

    @property
    def cons_node(self) -> "Node":
        return self._cons_node
