import torch

def edge_index_to_adj_matrix(edge_index):
    num_nodes = edge_index.max().item() + 1
        
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    row, col = edge_index
    adj[row, col] = 1.0
    
    return adj
