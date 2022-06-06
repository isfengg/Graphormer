from ogb.graphproppred import PygGraphPropPredDataset
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from functools import lru_cache
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from algos import floyd_warshall
import copy


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    # if len(edge_attr.size()) == 1:
    #     edge_attr = edge_attr[:, None]
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
    #     convert_to_single_emb(edge_attr) + 1
    # )

    shortest_path_result, path = floyd_warshall(adj.numpy())
    # max_dist = np.amax(shortest_path_result)
    # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([(N + 1) * (N + 1), 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    # item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos.view(N**2)
    item.degree = adj.long().sum(dim=1).view(-1)
    # item.edge_input = torch.from_numpy(edge_input).long()

    return item



class MyPygGraphPropPredDataset(PygGraphPropPredDataset):

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        return preprocess_item(item)
