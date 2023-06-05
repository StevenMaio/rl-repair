import torch
from torch import nn

from src.rl.graph import Graph
from src.rl.model import MultilayerPerceptron

from itertools import chain


def create_block(node_feat, neighbor_feat, edge_features):
    """
    Takes node_feat u, neighbor_feat matrix A, and edge_features B and
    returns the matrix whose i-th row is the vector (u, a_i, b_i), where
    a_i and b_i are the i-th row of A and B respectively.
    :param node_feat:
    :param neighbor_feat:
    :param edge_features:
    :return:
    """
    node_feat = node_feat.unsqueeze(0)
    node_feat = node_feat.expand(neighbor_feat.shape)
    return torch.cat((node_feat, neighbor_feat, edge_features), dim=1)


class GraphNeuralNetwork(nn.Module):
    _num_node_features: int
    _num_edge_features: int
    _num_iterations: int

    # the policy architecture
    _var_update_functions: nn.ModuleList
    _var_messenger_functions: nn.ModuleList
    _cons_update_functions: nn.ModuleList
    _cons_update_functions: nn.ModuleList

    def __init__(self,
                 params: "GnnParams"):
        super().__init__()

        num_node_features = params.num_node_features
        num_edge_features = params.num_edge_features
        num_iterations = params.num_gnn_iterations
        intermediate_layers = params.intermediate_layers
        hidden_layers = params.hidden_layers

        self._num_node_features = num_node_features
        self._num_edge_features = num_edge_features
        self._num_iterations = num_iterations

        # convert iteration_sizes and hidden_layers to iterables
        if isinstance(intermediate_layers, int):
            intermediate_layers = [intermediate_layers] * num_iterations
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers] * num_iterations
        assert (len(hidden_layers) == num_iterations)
        assert (len(intermediate_layers) == num_iterations)

        var_update_functions = []
        var_messenger_functions = []
        cons_update_functions = []
        cons_messenger_functions = []

        for input_size, output_size, hidden_layer in zip(chain((num_node_features,), intermediate_layers[:-1]),
                                                         intermediate_layers,
                                                         hidden_layers):
            var_update_functions.append(MultilayerPerceptron([2 * input_size,
                                                              hidden_layer,
                                                              output_size]))
            cons_update_functions.append(MultilayerPerceptron([2 * input_size,
                                                               hidden_layer,
                                                               output_size]))
            var_messenger_functions.append(MultilayerPerceptron([2 * input_size + num_edge_features,
                                                                 hidden_layer,
                                                                 input_size]))
            cons_messenger_functions.append(MultilayerPerceptron([2 * input_size + num_edge_features,
                                                                  hidden_layer,
                                                                  input_size]))

        self._var_update_functions = nn.ModuleList(var_update_functions)
        self._var_messenger_functions = nn.ModuleList(var_messenger_functions)
        self._cons_update_functions = nn.ModuleList(cons_update_functions)
        self._cons_messenger_functions = nn.ModuleList(cons_messenger_functions)

    def forward(self, graph: Graph):
        var_features = list(map(lambda u: u.features, graph.var_nodes))
        cons_features = list(map(lambda u: u.features, graph.cons_nodes))
        for iter_no in range(self._num_iterations):
            var_features, cons_features = self._forward_iteration(graph,
                                                                  var_features,
                                                                  cons_features,
                                                                  iter_no)
        return var_features, cons_features

    def _forward_iteration(self, graph, var_features, cons_features, iter_no):
        """
        Helper method that computes an update iteration in the graph neural network
        :param graph:
        :param var_features:
        :param cons_features:
        :param iter_no:
        :return:
        """
        updated_vars = []
        updated_cons = []
        for idx, (var_node, feat) in enumerate(zip(graph.var_nodes, var_features)):
            coefs = []
            neighbor_features = []
            for edge in var_node.edges:
                cons_node = edge.cons_node
                cons_id = cons_node.data_id
                coefs.append(edge.features)
                neighbor_features.append(cons_features[cons_id])
            if len(neighbor_features) > 0:
                neighbor_features = torch.stack(neighbor_features)
                coefs = torch.stack(coefs)
                msg_input = create_block(feat, neighbor_features, coefs)
                msg_output = self._var_messenger_functions[iter_no](msg_input).sum(dim=0)
                update_input = torch.cat((feat, msg_output))
            else:
                update_input = torch.cat((feat, torch.zeros_like(feat)))
            updated_vars.append(self._var_update_functions[iter_no](update_input))
        for idx, (cons_node, feat) in enumerate(zip(graph.cons_nodes, cons_features)):
            coefs = []
            neighbor_features = []
            for edge in cons_node.edges:
                var_node = edge.var_node
                var_id = var_node.data_id
                coefs.append(edge.features)
                neighbor_features.append(var_features[var_id])
            if len(neighbor_features) > 0:
                neighbor_features = torch.stack(neighbor_features)
                coefs = torch.stack(coefs)
                msg_input = create_block(feat, neighbor_features, coefs)
                msg_output = self._var_messenger_functions[iter_no](msg_input).sum(dim=0)
                update_input = torch.cat((feat, msg_output))
            else:
                update_input = torch.cat((feat, torch.zeros_like(feat)))
            updated_cons.append(self._cons_update_functions[iter_no](update_input))
        return updated_vars, updated_cons
