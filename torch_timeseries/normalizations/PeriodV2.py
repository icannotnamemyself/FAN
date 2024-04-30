


import torch
import torch.nn as nn
#  period_lens=[24*7, 24]
class PeriodV2(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in, period_lens=[12], ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.period_lens = period_lens
        
        self.period_len = 12
        
        
        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))
        
    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = MLP( seq_len, pred_len, enc_in, self.period_len, mode='std').float()

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        self.periods = []
        x_norm = input
        for P in self.period_lens:
            M = int(self.seq_len/P)
            x_windows =  x_norm.reshape(-1, P, M, self.enc_in)
            period = x_windows.mean(2, keepdim=True)
            self.periods.append(period)
            x_windows = x_windows - period
            x_norm = x_windows.reshape(-1, self.seq_len, self.enc_in)
        
        # trend normalize
        input = x_norm.reshape(bs, -1, self.period_len, dim)
        mean = torch.mean(input, dim=-2, keepdim=True)
        std = torch.std(input, dim=-2, keepdim=True)
        norm_input = (input - mean) / (std + self.epsilon)
        input = input.reshape(bs, len, dim)
        
        # statistic prediction
        mean_all = torch.mean(input, dim=1, keepdim=True)
        self.preds_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                        self.weight[1]
        self.preds_mean = self.preds_mean[:, -self.pred_len_new:, :]
        self.preds_std = self.model_std(std.squeeze(2), input)
        self.preds_std = self.preds_std[:, -self.pred_len_new:, :]
        
        return norm_input.reshape(bs, len, dim)

        
        # # trend normalize
        # input = input.reshape(bs, -1, self.period_len, dim)
        # mean = torch.mean(input, dim=-2, keepdim=True)
        # std = torch.std(input, dim=-2, keepdim=True)
        # norm_input = (input - mean) / (std + self.epsilon)
        # input = input.reshape(bs, len, dim)
        
        # # statistic prediction
        # mean_all = torch.mean(input, dim=1, keepdim=True)
        # self.preds_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
        #                 self.weight[1]
        # self.preds_mean = self.preds_mean[:, -self.pred_len_new:, :]
        # self.preds_std = self.model_std(std.squeeze(2), input)
        # self.preds_std = self.preds_std[:, -self.pred_len_new:, :]
        
        # # period normalize
        # x_norm = norm_input.reshape(bs, len, dim)
        # self.periods = []
        # for P in self.period_lens:
        #     M = int(self.seq_len/P)
        #     x_windows =  x_norm.reshape(-1, P, M, self.enc_in)
        #     period = x_windows.mean(2, keepdim=True)
        #     self.periods.append(period)
        #     x_windows = x_windows - period
        #     x_norm = x_windows.reshape(-1, self.seq_len, self.enc_in)
        
        # return x_norm

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape

        # trend denormalize
        input = input_norm.reshape(bs, -1, self.period_len, dim)
        output = input * (self.preds_std.unsqueeze(2)+ self.epsilon) + self.preds_mean.unsqueeze(2)
        input_norm = output.reshape(bs, len, dim)
        
        # # period denormalize 
        for P, period in zip(self.period_lens, self.periods ):
             M = int(len/P) + 1
             input_norm = input_norm + period.repeat(1,M , 1, 1)[:, :len, :, :].squeeze(2)
             
        # # # period denormalize 
        # for P, period in zip(self.period_lens, self.periods ):
        #      M = int(len/P) + 1
        #      input_norm = input_norm + period.repeat(1,M , 1, 1)[:, :len, :, :].squeeze(2)
             
        # # trend denormalize
        # input = input_norm.reshape(bs, -1, self.period_len, dim)
        # output = input * (self.preds_std.unsqueeze(2)+ self.epsilon) + self.preds_mean.unsqueeze(2)
        
        return output.reshape(bs, len, dim)
    
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
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)