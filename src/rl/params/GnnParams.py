DEFAULT_NUM_NODE_FEATURES = 10
DEFAULT_NUM_EDGE_FEATURES = 1
DEFAULT_INTERMEDIATE_LAYERS = 32
DEFAULT_HIDDEN_LAYERS = 32
DEFAULT_NUM_GNN_ITERATIONS = 2


class GnnParams:
    num_node_features: int = DEFAULT_NUM_NODE_FEATURES
    num_edge_features: int = DEFAULT_NUM_EDGE_FEATURES
    intermediate_layers: int = DEFAULT_INTERMEDIATE_LAYERS
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS
    num_gnn_iterations: int = DEFAULT_NUM_GNN_ITERATIONS

    def __init__(self,
                 num_node_features: int = DEFAULT_NUM_NODE_FEATURES,
                 num_edge_features: int = DEFAULT_NUM_EDGE_FEATURES,
                 intermediate_layers: int = DEFAULT_INTERMEDIATE_LAYERS,
                 hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
                 num_gnn_iterations: int = DEFAULT_NUM_GNN_ITERATIONS):
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.intermediate_layers = intermediate_layers
        self.hidden_layers = hidden_layers
        self.num_gnn_iterations = num_gnn_iterations
