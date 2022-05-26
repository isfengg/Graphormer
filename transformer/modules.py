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
    def __init__(self, nodes, degree, embedding_dim, layers):
        super().__init__()
        self.node_encoder = torch.nn.Embedding(nodes + 1, embedding_dim) # + 1 for specical token
        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim)
        self.graph_token = torch.nn.Embedding(1, embedding_dim)
        self.apply(lambda module: init_params(module, layers=layers)) # Refactor this shit
    
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

class GraphAttnBias(torch.nn.Module):
    def __init__(self, num_spatial, heads, layers):
        super().__init__()
        self.spatial_pos_encoder = torch.nn.Embedding(num_spatial, heads)
        self.apply(lambda module: init_params(module, layers=layers))
    
    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data['attn_bias'],
            batched_data["spatial_pos"],
            batched_data["x"]
        )

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.repeat(
            n_graph, self.num_heads, n_node+1, n_node+1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        return graph_attn_bias
        

