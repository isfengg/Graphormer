from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import IPython
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from algos import floyd_warshall

class Dataset(PygGraphPropPredDataset):
    def __init__(self):
        super().__init__("ogbg-molhiv")

    def find(self, idx):
        item = self[idx]
        return preprocess_item(item)

def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    shortest_path_result, _ = floyd_warshall(adj.numpy()) # O(n^3) time complexity for undirected unweighted graph? Easy solution right here.
    
    # edge feature here
    item.degree = adj.long().sum(dim=1).view(-1)
    item.spatial_pos = torch.from_numpy((shortest_path_result)).long()
    item.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float) # bias parameter for each SPD value with graph token
    
    return item

dataset = Dataset()
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
IPython.embed()