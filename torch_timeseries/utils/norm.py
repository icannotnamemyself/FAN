import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import FAConv,HeteroConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor, fill_diag
def hetero_directed_norm(edge_index, edge_weight=None, num_nodes=None,
              dtype=None):

    if isinstance(edge_index, SparseTensor):
        raise NotImplementedError("Operation of Sparse Tensor Not defined!")
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        
        # in degree of every node |N|
        in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

        # out degree of every node |N|
        out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        # nomalization
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)

        # source node out degree, target node in degree 
        return out_deg_inv_sqrt[row] * edge_weight * in_deg_inv_sqrt[col]



