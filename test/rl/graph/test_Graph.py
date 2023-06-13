"""
The tests test_gradient_ascent and test_gradient_ascent_with_mlp
are somewhat random. They could potentially fail, but hopefully that
won't happen.
"""

from unittest import TestCase

import torch

from src.rl.mip import EnhancedModel
from src.mip.model import *

from src.rl.graph import Graph
from src.rl.params import GnnParams
from src.rl.graph.Node import FeatIdx
from src.rl.architecture import GraphNeuralNetwork, MultilayerPerceptron


class TestGraph(TestCase):

    def test_construct_mip_graph(self):
        model = EnhancedModel()

        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.INTEGER)
        model.add_var(variable_type=VarType.CONTINUOUS)

        model.add_constraint([x_id, y_id],
                             [1.0, 2.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.LE)
        model.init()
        G = Graph(model)
        G.update(model)

        # test variable node features
        x_node = G.var_nodes[0]
        y_node = G.var_nodes[1]
        z_node = G.var_nodes[2]

        # construct features
        expected_x_feat = torch.zeros(GnnParams.num_node_features)
        expected_x_feat[FeatIdx.IS_BINARY] = 1.0
        expected_x_feat[FeatIdx.LOCAL_DOMAIN_SIZE] = 1.0

        expected_y_feat = torch.zeros(GnnParams.num_node_features)
        expected_y_feat[FeatIdx.IS_INTEGER] = 1.0
        expected_y_feat[FeatIdx.LOCAL_DOMAIN_SIZE] = 1.0

        expected_z_feat = torch.zeros(GnnParams.num_node_features)
        expected_z_feat[FeatIdx.IS_CONTINUOUS] = 1.0
        expected_z_feat[FeatIdx.LOCAL_DOMAIN_SIZE] = 1.0

        self.assertTrue(torch.allclose(expected_x_feat, x_node.features))
        self.assertTrue(torch.allclose(expected_y_feat, y_node.features))
        self.assertTrue(torch.allclose(expected_z_feat, z_node.features))

        # test constraint node features
        c0_node = G.cons_nodes[0]
        c1_node = G.cons_nodes[1]

        # expected c1 feautures
        expected_c0_feat = torch.zeros(GnnParams.num_node_features)
        expected_c0_feat[FeatIdx.IS_EQ_CONSTRAINT] = 1.0
        expected_c0_feat[FeatIdx.NUM_VARIABLES] = 1.0

        # expected c1 feautures
        expected_c1_feat = torch.zeros(GnnParams.num_node_features)
        expected_c1_feat[FeatIdx.IS_LE_CONSTRAINT] = 1.0
        expected_c1_feat[FeatIdx.NUM_VARIABLES] = 1.0

        self.assertTrue(torch.allclose(expected_c0_feat, c0_node.features))
        self.assertTrue(torch.allclose(expected_c1_feat, c1_node.features))

        c0_e0 = c0_node.edges[0]
        c0_e1 = c0_node.edges[1]
        self.assertEqual(1.0, c0_e0.features.item())
        self.assertEqual(2.0, c0_e1.features.item())

    def test_graph_neural_network(self):
        # instantiate mip instance and graph
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.INTEGER)
        model.add_var(variable_type=VarType.CONTINUOUS)
        model.add_constraint([x_id, y_id],
                             [1.0, 2.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.LE)
        model.init()
        G = Graph(model)
        gnn = GraphNeuralNetwork(GnnParams)

        var_features, cons_features = gnn(G)
        self.assertEqual(3, len(var_features))
        self.assertEqual(2, len(cons_features))

        h_new = var_features[0]
        self.assertEqual(torch.Size([GnnParams.intermediate_layers]),
                         h_new.shape)

    def test_gradient_ascent(self):
        # instantiate mip instance and graph
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.INTEGER)
        model.add_var(variable_type=VarType.CONTINUOUS)
        model.add_constraint([x_id, y_id],
                             [1.0, 2.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.LE)
        model.init()
        G = Graph(model)
        gnn = GraphNeuralNetwork(GnnParams)

        var_features, cons_features = gnn(G)

        first_output = var_features[0].pow(2).sum()

        first_output.backward()
        params = gnn.parameters()
        with torch.no_grad():
            # add the gradient to the params whenever possible
            for p in params:
                g = p.grad
                if g is not None:
                    torch.add(p, g, out=p)

            # compute the next convolution and compute the new value
            var_features, cons_features = gnn(G)
            output_after_grad_ascent = var_features[0].pow(2).sum()
        self.assertLess(first_output, output_after_grad_ascent)

    def test_gradient_ascent_with_mlp(self):
        # instantiate mip instance and graph
        model = Model()
        x_id: int = model.add_var(variable_type=VarType.BINARY)
        y_id: int = model.add_var(variable_type=VarType.INTEGER)
        model.add_var(variable_type=VarType.CONTINUOUS)
        model.add_constraint([x_id, y_id],
                             [1.0, 2.0],
                             1.0,
                             Sense.EQ)
        model.add_constraint([x_id, y_id],
                             [1.0, 1.0],
                             1.0,
                             Sense.LE)
        model.init()
        G = Graph(model)
        gnn = GraphNeuralNetwork(GnnParams)

        var_features, cons_features = gnn(G)

        scoring_function = MultilayerPerceptron([GnnParams.intermediate_layers,
                                                 32,
                                                 1])

        first_output = scoring_function(var_features[0])

        first_output.backward()
        params = gnn.parameters()
        with torch.no_grad():
            # add the gradient to the params whenever possible
            for p in params:
                g = p.grad
                if g is not None:
                    torch.add(p, g, out=p)

            # compute the next convolution and compute the new value
            var_features, cons_features = gnn(G)
            output_after_grad_ascent = scoring_function(var_features[0])
        print(f'first_pass={first_output.item()}\nsecond_pass={output_after_grad_ascent.item()}')
        self.assertLess(first_output.item(), output_after_grad_ascent.item())
