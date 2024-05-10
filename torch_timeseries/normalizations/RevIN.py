import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, n_series, affine):
        super().__init__()
        self.affine = affine
        if affine: # affine: use affine layers or not
            self.gamma = nn.Parameter(torch.ones(n_series)) # n_series: number of series
            self.beta = nn.Parameter(torch.zeros(n_series))
        else:
            self.gamma, self.beta = 1, 0
    
    def normalize(self, batch_x, dec_inp=None):
        # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
        self.preget(batch_x)
        batch_x = self.forward_process(batch_x)
        dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
        return batch_x, dec_inp
    
    def loss(self, true):
        return 0

    def denormalize(self, batch_x):
        # batch_x: B*H*D (forecasts)
        batch_y = self.inverse_process(batch_x)
        return batch_y


    def preget(self, batch_x):
        # (B, T, N)
        self.avg = torch.mean(batch_x, axis=1, keepdim=True).detach() # b*1*d
        self.var = torch.var(batch_x, axis=1, keepdim=True).detach()  # b*1*d

    def forward_process(self, batch_input):
        temp = (batch_input - self.avg)/torch.sqrt(self.var + 1e-8)
        if self.affine:
            return temp.mul(self.gamma) + self.beta
        else: 
            return temp

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.var + 1e-8) + self.avg
    
    def forward(self, batch_x, mode='n', dec=None):
        if mode == 'n':
            return self.normalize(batch_x, dec)
        elif mode =='d':
            return self.denormalize(batch_x)
            


