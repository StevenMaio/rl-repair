import torch


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
    node_feat = node_feat.expand((neighbor_feat.shape[0], node_feat.shape[1]))
    return torch.cat((node_feat, neighbor_feat, edge_features), dim=1)
