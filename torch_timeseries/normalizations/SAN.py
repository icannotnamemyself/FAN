import torch
import torch.nn as nn


class SAN(nn.Module):
    def __init__(self, seq_len, pred_len, period_len, enc_in,station_type='adaptive'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.channels = enc_in 
        self.enc_in = enc_in 
        # configs.enc_in if configs.features == 'M' else 1
        self.station_type = station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        seq_len = self.seq_len // self.period_len
        enc_in = self.enc_in
        pred_len = self.pred_len_new
        self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
        self.model_std = MLP( seq_len, pred_len, enc_in, self.period_len, mode='std').float()

    def normalize(self, input):
        # (B, T, N)
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            std = torch.std(input, dim=-2, keepdim=True)
            norm_input = (input - mean) / (std + self.epsilon)
            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)
            outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                           self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input)

            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)

            return norm_input.reshape(bs, len, dim), outputs[:, -self.pred_len_new:, :]

        else:
            return input, None

    def denormalize(self, input, station_pred):
        # input:  (B, O, N)
        # station_pred: outputs of normalize 
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            # mean = station_pred[:, :, :self.channels].unsqueeze(2)
            # std = station_pred[:, :, self.channels:].unsqueeze(2)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, len, dim)
        else:
            return input
    def forward(self, batch_x, mode='n', station_pred=None):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x, station_pred)


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