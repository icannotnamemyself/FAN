


import torch
import torch.nn as nn

# use less
class SNv3(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 

    def normalize(self, batch_x):
        # (B, T, N)
        diff = torch.diff(batch_x, dim=1)
        padding = batch_x[:, 0, :].unsqueeze(1)
        x_norm = torch.cat([padding, diff], dim=1)
        return x_norm


    def denormalize(self, input):
        # input:  (B, O, N)
        # station_pred: outputs of normalize 
        bs, len, dim = input.shape
        pred = torch.cumsum(input, dim=1) # (B, O, N)
        return pred
    
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)
