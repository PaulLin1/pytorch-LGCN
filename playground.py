from torch_geometric.datasets import KarateClub, Planetoid

from processing import edge_index_to_adj_matrix
from models.LGCL import LGCL
 
# dataset = KarateClub()
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]
len(data.x)
# adj_matrix = edge_index_to_adj_matrix(data.edge_index)
# node_features = dataset.x

# z = LGCL(node_features, adj_matrix, 4, 5, 6)

# print(z)