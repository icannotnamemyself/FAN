


import torch
import torch.nn as nn
#  period_lens=[24*7, 24]
class PeriodV1(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in, period_lens=[24*7, 24]):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.period_lens = period_lens
        self.periods = []
            

    def normalize(self, batch_x):
        # (B, T, N)
        x_norm = batch_x
        self.periods = []
        for P in self.period_lens:
            M = int(self.seq_len/P)
            x_windows =  x_norm.reshape(-1, P, M, self.enc_in)
            period = x_windows.mean(2, keepdim=True).detach()
            self.periods.append(period)
            x_windows = x_windows -period
            x_norm = x_windows.reshape(-1, self.seq_len, self.enc_in)
        
        return x_norm

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        
        for P, period in zip(self.period_lens, self.periods ):
             M = int(len/P) + 1
             input_norm = input_norm + period.repeat(1,M , 1, 1)[:, :len, :, :].squeeze(2)
            # input_windows = input_norm.reshape(-1, P, M, self.enc_in)
        return input_norm.reshape(bs, len, dim)
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)

