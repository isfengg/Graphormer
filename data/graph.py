from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import torch.distributed as dist
import torch

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

    # edge feature here
    item.degree = adj.long().sum(dim=1).view(-1)

    return item

dataset = Dataset()
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)