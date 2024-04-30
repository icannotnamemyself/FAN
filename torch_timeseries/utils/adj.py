import torch
import numpy as np
from scipy.sparse import coo_matrix

def adj_to_edge_index_weight(adj_matrix):
    adj_coo = coo_matrix(adj_matrix)
    
    src = adj_coo.row
    dst = adj_coo.col
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    weight = torch.tensor(adj_coo.data, dtype=torch.float)
    
    return edge_index, weight
