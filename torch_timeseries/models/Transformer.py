import torch
import numpy as np
import torch.nn.functional as F






torch.nn.MultiheadAttention()
class MultiHeadsAttention(torch.nn.Module):
    
    
    def __init__(self, embed_dim:int, heads:int = 3 , q_dim=None, k_dim=None, v_dim=None) -> None:
        super().__init__()
        
        self.queries_projections = torch.nn.ModuleList()
        self.keys_projections = torch.nn.ModuleList()
        self.values_projections = torch.nn.ModuleList()
        self.heads = 3
        self.embed_dim = embed_dim
        
        if q_dim is None:
            self.q_dim = embed_dim
        if k_dim is None:
            self.k_dim = embed_dim
        if v_dim is None:
            self.v_dim = embed_dim

        for i in range(heads):
            self.queries_projections.append(torch.nn.Linear( self.q_dim, embed_dim))
            self.keys_projections.append(torch.nn.Linear( self.k_dim, embed_dim))
            self.values_projections.append(torch.nn.Linear( self.v_dim, embed_dim))
    
    
    def forward(self, q, k, v):
        queries = []
        keys = []
        values = []
        for i in range(self.heads):
            queries.append(self.queries_projections(q))
            keys.append(self.keys_projections(q))
            values.append(self.values_projections(q))
        
        
                
        
        


class Transformer(torch.nn.Module):
    pass

