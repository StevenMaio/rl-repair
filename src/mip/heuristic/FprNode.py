from typing import List


class FprNode:
    _parent: "FprNode"
    _left: "FprNode"
    _right: "FprNode"
    _domain_changes: List["DomainChange"]
    _visited: bool
    _fixed_var_id: int
    _depth: int
    _id: int

    # static
    _node_counter: int

    def __init__(self,
                 parent: "FprNode" = None,
                 fixed_var_id: int = -1,
                 fixing: "DomainChange" = None):
        self._parent = parent
        self._left = None
        self._right = None
        if fixing is None:
            self._domain_changes = []
        else:
            self._domain_changes = [fixing]
        self._visited = False
        self._fixed_var_id = fixed_var_id
        if parent is None:
            self._depth = 0
        else:
            self._depth = parent.depth + 1
        self._id = self._node_counter
        self._node_counter += 1

    @property
    def parent(self) -> "FprNode":
        return self._parent

    @parent.setter
    def parent(self, new_value: "FprNode"):
        self._parent = new_value

    @property
    def left(self) -> "FprNode":
        return self._left

    @left.setter
    def left(self, new_value: "FprNode"):
        self._left = new_value

    @property
    def right(self) -> "FprNode":
        return self._right

    @right.setter
    def right(self, new_value: "FprNode"):
        self._right = new_value

    @property
    def domain_changes(self):
        return self._domain_changes

    @property
    def visited(self) -> bool:
        return self._visited

    @visited.setter
    def visited(self, new_value: bool):
        self._visited = new_value

    @property
    def fixed_var_id(self) -> int:
        return self._fixed_var_id

    @fixed_var_id.setter
    def fixed_var_id(self, new_value: int):
        self._fixed_var_id = new_value

    @property
    def depth(self):
        return self._depth

    @property
    def id(self) -> int:
        return self._id


FprNode._node_counter = 0
