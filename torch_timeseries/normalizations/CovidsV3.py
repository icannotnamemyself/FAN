import torch
import torch.nn as nn


def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2):
    k = mu1.size(1)
    
    Sigma2_inv = torch.linalg.inv(Sigma2)
    
    tr_term = torch.einsum('bij,bjk->bi', Sigma2_inv, Sigma1)
    
    mu_diff = mu2 - mu1
    mu_term = torch.einsum('bi,bij,bj->b', mu_diff, Sigma2_inv, mu_diff)
    
    det_term = torch.log(torch.linalg.det(Sigma2) / torch.linalg.det(Sigma1))
    
    kl_div = 0.5 * (tr_term + mu_term - k + det_term)
    
    return kl_div.sum()



class CovidsV3(nn.Module):
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

        self.model_U = nn.Sequential(
            nn.Linear(enc_in, 256),
            nn.ReLU(),
            nn.Linear(256, enc_in),
        )
        self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean')
        self.model_U2 = MLP( seq_len, pred_len, enc_in, self.period_len, mode='U2')

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        T = self.period_len
        M = int(len / T)
        input = input.reshape(bs, -1, self.period_len, dim) # (B, M, T, N)
        mean = torch.mean(input, dim=-2, keepdim=True) # (B, M, 1, N)
        
        # get x_norm
        x_center = input - mean # (B, M, T, N)
        Sigma = torch.einsum('bmtv,bmtn->bmvn', x_center, x_center) / (T - 1) # (B, M, N, N)
        U = self.model_U(Sigma)
        x_norm = torch.einsum('bmvn,bmtn->bmtv', U, x_center)
        
        # for loss1
        x_norm_mean = torch.mean(x_norm, dim=-2, keepdim=True)
        self.x_norm_mean = x_norm_mean
        self.Sigma_norm = torch.einsum('bmtv,bmtn->bmvn', x_norm - x_norm_mean, x_norm - x_norm_mean) / (T - 1) 
        # traces = torch.diagonal(Sigma_norm, dim1=-2, dim2=-1).sum(-1)
        # KL_loss = 0.5 * (-Sigma_norm.det() - dim + traces + ((x_norm_mean.reshape(bs, M, dim))**2).sum(-1).shape).sum()
        # self.kl_loss1 = kl_divergence_gaussian(x_norm_mean, Sigma_norm, torch.zeros(dim), torch.eye(dim))
        
        # predict U2
        self.U2 = self.model_U2(Sigma, Sigma, 'U2')

        # predict statistics
        input = input.reshape(bs, len, dim)
        mean_all = torch.mean(input, dim=1, keepdim=True)
        outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all, 'mean') * self.weight[0] + mean_all * \
                        self.weight[1]
        # outputs_sigma = self.model_sigma(Sigma, input, mode='sigma')

        self.preds_mean  = outputs_mean[:, -self.pred_len_new:, :]  # (B, M', N)
        # self.preds_sigma  = outputs_sigma[:, -self.pred_len_new:, :, :] # (B, M', N, N)
        # import pdb;pdb.set_trace()
        

        # outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
        # self.preds_mean = outputs[:, -self.pred_len_new:, :]
        return x_norm.reshape(bs, len, dim)

    def denormalize(self, output):
        # input:  (B, O, N)
        # import pdb;pdb.set_trace()
        
        # station_pred: outputs of normalize 
        bs, len, dim = output.shape

        output = output.reshape(bs, -1, self.period_len, dim) # (B, M', T, N)

        output = torch.einsum('bmvn,bmtn->bmtv', self.U2, output) + self.preds_mean.unsqueeze(-2)
        
        self.output_mean = torch.mean(output, dim=-2, keepdim=True) # (B, M, 1, N)
        output_center = output - self.output_mean # (B, M, T, N)
        self.output_Sigma = torch.einsum('bmtv,bmtn->bmvn', output_center, output_center) / (self.period_len - 1) # (B, M, N, N)
        
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
        if mode == 'U2':
            self.input = nn.Sequential(
                nn.Linear(self.seq_len, 32),
                nn.ReLU(),
                nn.Linear(32, self.pred_len),
            )
            # self.final_activation = nn.Identity()
            # self.input_raw1 = nn.Linear(int(seq_len), 32)
            # self.input_raw2 = nn.Linear(int(seq_len), 32)
            # self.activation = nn.Tanh()
            # self.output = nn.Linear(128, self.pred_len) 

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
        
        if self.mode == 'U2':
            B, M, N, N = x_raw.shape
            x, x_raw = x.permute(0,2,3,1), x_raw.permute(0,2,3,1) # (B, N, N, M)
            x = self.input(x)
            # x_raw1 = self.input_raw1(x_raw)
            # x_raw2 = self.input_raw2(x_raw)
            # x = torch.cat([x_raw1, x_raw2], dim=-1)  # (B, N, N, 1024)
            # x = self.output(self.activation(x)) # (B, N, N, O)
            # x = self.final_activation(x) 
            return x.permute(0, 3, 1, 2) # (B, O, N, N)
            
            # x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)