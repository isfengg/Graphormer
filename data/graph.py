from ogb.graphproppred import PygGraphPropPredDataset
import torch
import numpy as np
from functools import lru_cache
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from algos import floyd_warshall


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True


    shortest_path_result, path = floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.degree = adj.long().sum(dim=1).view(-1)

    return item



class MyPygGraphPropPredDataset(PygGraphPropPredDataset):

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        return preprocess_item(item)
