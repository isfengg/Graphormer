import torch
import math

def init_params(module, layers):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GraphNodeFeatures(torch.nn.Module):
    def __init__(self, heads, nodes, degree, embedding_dim, layers):
        super().__init__()
        self.node_encoder = torch.nn.Embedding(nodes, embedding_dim)
        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim)
        self.graph_token = torch.nn.Embedding(1, embedding_dim)
        self.apply(lambda module: init_params(module, layers=layers))
    
    def forward(self, batched_data):
        x, degree = (
            batched_data['x'],
            batched_data['degree']
        )
        
        n_graph, n_node = x.size()[:2]
        node_feature = self.node_encoder(x).sum(dim=-2)
        node_feature = (
            node_feature
            + self.degree_encoder(degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature
