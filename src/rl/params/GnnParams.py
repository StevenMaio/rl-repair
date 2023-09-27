DEFAULT_NUM_VAR_NODE_FEATURES = 11
DEFAULT_NUM_CONS_NODE_FEATURES = 10
DEFAULT_NUM_EDGE_FEATURES = 3
DEFAULT_INTERMEDIATE_LAYERS = 32
DEFAULT_HIDDEN_LAYERS = 32
DEFAULT_NUM_GNN_ITERATIONS = 2
DEFAULT_ADD_BATCH_NORM_PARAMS = True
DEFAULT_ADDITIONAL_VAR_SCORING_FEATURES = 1


class GnnParams:
    num_var_node_features: int = DEFAULT_NUM_VAR_NODE_FEATURES
    num_cons_node_features: int = DEFAULT_NUM_CONS_NODE_FEATURES
    num_edge_features: int = DEFAULT_NUM_EDGE_FEATURES
    intermediate_layers: int = DEFAULT_INTERMEDIATE_LAYERS
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS
    num_gnn_iterations: int = DEFAULT_NUM_GNN_ITERATIONS
    add_batch_norm_params: bool = DEFAULT_ADD_BATCH_NORM_PARAMS
    additional_var_scoring_features: int = DEFAULT_ADDITIONAL_VAR_SCORING_FEATURES

    def __init__(self,
                 num_var_node_features: int = DEFAULT_NUM_VAR_NODE_FEATURES,
                 num_cons_node_features: int = DEFAULT_NUM_CONS_NODE_FEATURES,
                 num_edge_features: int = DEFAULT_NUM_EDGE_FEATURES,
                 intermediate_layers: int = DEFAULT_INTERMEDIATE_LAYERS, hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
                 num_gnn_iterations: int = DEFAULT_NUM_GNN_ITERATIONS,
                 add_batch_norm_params: bool = DEFAULT_ADD_BATCH_NORM_PARAMS,
                 additional_var_scoring_features: int = DEFAULT_ADDITIONAL_VAR_SCORING_FEATURES):
        self.num_var_node_features = num_var_node_features
        self.num_cons_node_features = num_cons_node_features
        self.num_edge_features = num_edge_features
        self.intermediate_layers = intermediate_layers
        self.hidden_layers = hidden_layers
        self.num_gnn_iterations = num_gnn_iterations
        self.add_batch_norm_params = add_batch_norm_params
        self.additional_var_scoring_features = additional_var_scoring_features
