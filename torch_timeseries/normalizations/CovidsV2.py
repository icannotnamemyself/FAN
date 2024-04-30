import torch
import torch.nn as nn


class CovidsV2(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, period_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in 
        self.enc_in = enc_in 

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean')
        self.model_sigma = MLP( seq_len, pred_len, enc_in, self.period_len, mode='sigma')

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        T = self.period_len
        input = input.reshape(bs, -1, self.period_len, dim) # (B, M, T, N)
        mean = torch.mean(input, dim=-2, keepdim=True) # (B, M, 1, N)
        # std = torch.std(input, dim=-2, keepdim=True) # (B, M, 1, N)
        
        x_center = input - mean # (B, M, T, N)
        Sigma = torch.einsum('bmtv,bmtn->bmvn', x_center, x_center) / (T - 1) # (B, M, N, N)
        U, S, V = torch.linalg.svd(Sigma.detach(), full_matrices=False) 
        # U = U.detach()
        # S = S.detach()
        # V = V.detach()

        x_norm = (1/torch.sqrt(S+self.epsilon)).unsqueeze(-2) * torch.einsum('bmvn,bmtn->bmtv', U, x_center)
        # predict statistics
        input = input.reshape(bs, len, dim)
        mean_all = torch.mean(input, dim=1, keepdim=True)
        outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all, 'mean') * self.weight[0] + mean_all * \
                        self.weight[1]
        outputs_sigma = self.model_sigma(Sigma, input, mode='sigma')

        self.preds_mean  = outputs_mean[:, -self.pred_len_new:, :]  # (B, M', N)
        self.preds_sigma  = outputs_sigma[:, -self.pred_len_new:, :, :] # (B, M', N, N)
        # import pdb;pdb.set_trace()
        

        # outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
        # self.preds_mean = outputs[:, -self.pred_len_new:, :]
        return x_norm.reshape(bs, len, dim)

    def denormalize(self, input):
        # input:  (B, O, N)
        # import pdb;pdb.set_trace()
        
        # station_pred: outputs of normalize 
        bs, len, dim = input.shape
        input = input.reshape(bs, -1, self.period_len, dim) # (B, M', T, N)
        U, S, V = torch.linalg.svd(self.preds_sigma.detach()  , full_matrices=False) 
        # U = U.detach()
        # S = S.detach()
        # V = V.detach()
        # mean = station_pred[:, :, :self.channels].unsqueeze(2)
        # std = station_pred[:, :, self.channels:].unsqueeze(2)
        
        x_norm = torch.einsum('bmvn,bmtn->bmtv', V, torch.sqrt(S).unsqueeze(-2) * input)
        
        # import pdb;pdb.set_trace()
        # x_norm = V *   # (B, M', T, N)
        output = x_norm + self.preds_mean.unsqueeze(-2) # (B, M', T, N)
        return output.reshape(bs, len, dim) # (B,O, N)
    
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)


class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, period_len, mode):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.period_len = period_len
        self.mode = mode
        if mode == 'sigma':
            self.final_activation = nn.ReLU()
            self.input = nn.Linear(self.seq_len, 512)
            self.input_raw1 = nn.Linear(self.period_len, 512)
            self.input_raw2 = nn.Linear(self.period_len, 512)
            self.activation = nn.Tanh()
            self.output = nn.Linear(1024, self.pred_len) 

        else:
            self.final_activation = nn.Identity()
            self.input = nn.Linear(self.seq_len, 512)
            self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
            self.activation = nn.ReLU()
            self.output = nn.Linear(1024, self.pred_len)
            
        # self.input = nn.Linear(self.seq_len, 512)
        # self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        # self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        # self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw, mode):
        
        if self.mode == 'sigma':
            bs, len, dim = x_raw.shape
            x_raw = x_raw.reshape(bs, -1, self.period_len, dim) # (B, M, T, N)
            x, x_raw = x.permute(0, 3, 2, 1), x_raw.permute(0, 1,3, 2) # (B, N, N, M),  (B, M, N, T)

            x_raw1 = self.input_raw1(x_raw)
            x_raw2 = self.input_raw2(x_raw)
            x_raw = torch.einsum('bmvt,bmqt->bvqt' , x_raw1, x_raw2) # (B, N, N, 512)
            x = self.input(x) # (B, N, N, 512)
            # x_raw = self.input_raw(x_raw)
            x = torch.cat([x, x_raw], dim=-1)  # (B, N, N, 1024)
            x = self.output(self.activation(x)) # (B, N, N, O)
            x = self.final_activation(x)
            return x.permute(0, 3, 1, 2) # (B, O, N, N)
            
            # x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)