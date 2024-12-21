import torch
import torch.nn as nn

# g and c are the respective functions from the paper
def g(node, all_node_features, adj_matrix, k):    
    neighbors = adj_matrix[node]
    neighbor_indices = torch.where(neighbors == 1)[0]
    neighbors_features = all_node_features[neighbor_indices]
    if len(neighbors_features) < k:
        neighbors_features = torch.nn.functional.pad(neighbors_features, pad=(0, 0, 0, k - len(neighbors_features)))
    
    top_k_features, _ = torch.topk(neighbors_features, k=k, dim=0)
    self_and_top_k_features = torch.cat([all_node_features[node].unsqueeze(0), top_k_features], dim=0)
    
    return self_and_top_k_features

def c(selected_features, input_dim, output_dim, k):
    conv1d = torch.nn.Conv1d(
        in_channels=input_dim, 
        out_channels=output_dim, 
        kernel_size=k+1
    )

    conv_res = conv1d(selected_features)
    return conv_res

class LGCL(nn.Module):
    def __init__(self, node_features, adj_matrix, k, input_dim, output_dim):
        super().__init__()

        self.node_features = node_features
        self.adj_matrix = adj_matrix
        self.k = k
        self.input_dim = input_dim
        self.output_dim = output_dim

    def init_params(self, layer_type):
        pass

    def forward(self, x):
        selected_features = torch.stack([g(i, self.node_features, self.adj_matrix, self.k) for i in range(len(self.node_features))])
        return c(selected_features, self.input_dim, self.output_dim, self.k)

class LGCN(nn.Module):
    def __init__(self, node_features, adj_matrix, k, input_dim, output_dim):
        super().__init__()

        self.node_features = node_features
        self.adj_matrix = adj_matrix
        self.k = k
        self.input_dim = input_dim
        self.output_dim = output_dim

    def init_params(self, layer_type):
        pass
    
    def forward(self, x):
        selected_features = torch.stack([g(i, self.node_features, self.adj_matrix, self.k) for i in range(len(self.node_features))])
        return c(selected_features, self.input_dim, self.output_dim, self.k)
