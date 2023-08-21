"""
The second implementation of the graph neural network. This version splits batch
normalization of the node representations into variable nodes and constraint nodes.
"""
import torch
from torch import nn

from .utils import create_block
from src.rl.graph import Graph
from src.rl.architecture import MultilayerPerceptron

from typing import List


class GraphNeuralNetworkV2(nn.Module):
    _var_node_dimensions: List[int]
    _cons_node_dimensions: List[int]
    _num_edge_features: int
    _num_iterations: int

    # the policy architecture
    _var_update_functions: nn.ModuleList
    _var_messenger_functions: nn.ModuleList
    _var_init_batch_norms: nn.BatchNorm1d
    _var_batch_normalizations: nn.ModuleList

    _cons_update_functions: nn.ModuleList
    _cons_update_functions: nn.ModuleList
    _cons_init_batch_norms: nn.BatchNorm1d
    _cons_batch_normalizations: nn.ModuleList

    def __init__(self,
                 params: "GnnParams"):
        super().__init__()

        num_var_node_features = params.num_var_node_features
        num_cons_node_features = params.num_cons_node_features
        num_edge_features = params.num_edge_features
        num_iterations = params.num_gnn_iterations
        intermediate_layers = params.intermediate_layers
        hidden_layers = params.hidden_layers

        # convert iteration_sizes and hidden_layers to iterables
        if isinstance(intermediate_layers, int):
            intermediate_layers = [intermediate_layers] * num_iterations
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers] * num_iterations
        assert (len(hidden_layers) == num_iterations)
        assert (len(intermediate_layers) == num_iterations)

        self._var_node_dimensions = [num_var_node_features] + intermediate_layers
        self._cons_node_dimensions = [num_cons_node_features] + intermediate_layers
        self._num_edge_features = num_edge_features
        self._num_iterations = num_iterations

        var_update_functions = []
        var_messenger_functions = []
        cons_update_functions = []
        cons_messenger_functions = []
        var_batch_normalizations = []
        cons_batch_normalizations = []

        for var_input_size, cons_input_size, output_size, hidden_layer in zip(
                self._var_node_dimensions[:-1],
                self._cons_node_dimensions[:-1],
                intermediate_layers,
                hidden_layers):
            msg_input_size = var_input_size + cons_input_size + num_edge_features
            var_update_functions.append(MultilayerPerceptron([var_input_size + output_size,
                                                              hidden_layer,
                                                              output_size]))
            cons_update_functions.append(MultilayerPerceptron([cons_input_size + output_size,
                                                               hidden_layer,
                                                               output_size]))
            var_messenger_functions.append(MultilayerPerceptron([msg_input_size,
                                                                 hidden_layer,
                                                                 output_size]))
            cons_messenger_functions.append(MultilayerPerceptron([msg_input_size,
                                                                  hidden_layer,
                                                                  output_size]))
            var_batch_normalizations.append(nn.BatchNorm1d(output_size,
                                                           affine=params.add_batch_norm_params))
            cons_batch_normalizations.append(nn.BatchNorm1d(output_size,
                                                            affine=params.add_batch_norm_params))

        self._var_update_functions = nn.ModuleList(var_update_functions)
        self._var_messenger_functions = nn.ModuleList(var_messenger_functions)
        self._cons_update_functions = nn.ModuleList(cons_update_functions)
        self._cons_messenger_functions = nn.ModuleList(cons_messenger_functions)
        self._var_batch_normalizations = nn.ModuleList(var_batch_normalizations)
        self._cons_batch_normalizations = nn.ModuleList(cons_batch_normalizations)

    def forward(self, graph: Graph):
        var_features = torch.stack(list(map(lambda u: u.features, graph.var_nodes)))
        cons_features = torch.stack(list(map(lambda u: u.features, graph.cons_nodes)))
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
        # variable node update computation
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
                update_input = torch.cat((feat, torch.zeros(self._cons_node_dimensions[iter_no + 1])))
            updated_vars.append(self._var_update_functions[iter_no](update_input))
            # constraint node update computation
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
                msg_output = self._cons_messenger_functions[iter_no](msg_input).sum(dim=0)
                update_input = torch.cat((feat, msg_output))
            else:
                update_input = torch.cat((feat, torch.zeros(self._var_node_dimensions[iter_no + 1])))
            updated_cons.append(self._cons_update_functions[iter_no](update_input))
        updated_vars = torch.stack(updated_vars)
        updated_cons = torch.stack(updated_cons)
        updated_vars = self._var_batch_normalizations[iter_no](updated_vars)
        updated_cons = self._cons_batch_normalizations[iter_no](updated_cons)
        return updated_vars, updated_cons
