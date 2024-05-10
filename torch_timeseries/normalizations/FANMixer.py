


import time
import torch
import torch.nn as nn


from torch_timeseries.models.TSMixer import TSMixer


def noise_freq_part(x, k):
    # freq normalization
    # start = time.time()
    xf = torch.fft.fft(x, dim=1)
    k_values = torch.topk(xf.abs(), k, dim = 1)  
    indices = k_values.indices


    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask
    
    x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered

def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
        
    k_values = torch.topk(xf.abs(), k, dim = 1)  
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered


def frl(x, k):
    # x: (BxNxL) The batchified multivariate time series input
    # k: hyper parameter selecting k largest magnitude frequencies
    
    # applying fourier transform to each series O(Llog(L))
    z = torch.fft.rfft(x, dim=2)
    
    # find top k indices O(L + k)
    ks = torch.topk(z.abs(), k, dim = 1)  
    top_k_indices = ks.indices

    # top-k-pass filter O(L + k)
    mask = torch.zeros_like(z)
    mask.scatter_(1, top_k_indices, 1)
    z_m = z * mask

    # applying inverse fourier transform to each fourier series O(Llog(L))
    x_m = torch.fft.irfft(z_m, dim=2).real
    x_n = x - x_m
    return x_n


def low_freq_part(x, k):
    # freq normalization
    # start = time.time()
    xf = torch.fft.fft(x, dim=1)
    
    # 获取频率最高的k个分量
    low_freq = xf[:, :k, :]
    
    # 用零填充其余部分
    padding = torch.zeros_like(xf)
    padding[:, :k, :] = low_freq
    
    # 进行逆傅里叶变换
    x_filtered = torch.fft.ifft(padding, dim=1).real.float()
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered

def high_freq_part(x, k):
    # freq normalization
    # start = time.time()
    xf = torch.fft.fft(x, dim=1)
    
    # 获取频率最高的k个分量
    high_freq = xf[:, -k:, :]
    
    # 用零填充其余部分
    padding = torch.zeros_like(xf)
    padding[:, -k:, :] = high_freq
    
    # 进行逆傅里叶变换
    x_filtered = torch.fft.ifft(padding, dim=1).real.float()
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered


class FANMixer(nn.Module):
    """FAN first substract bottom k frequecy component from the original series
      

    Args:
        nn (_type_): _description_
    """
    def __init__(self,  seq_len, pred_len, enc_in, freq_topk = 20, period_len=12 , SAN=False, rfft=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk )
        self.period_len = period_len
        self.SAN = SAN
        self.rfft = rfft
        
        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))
        
    def _build_model(self):
        if self.SAN:
            seq_len = self.seq_len // self.period_len
            enc_in = self.enc_in
            pred_len = self.pred_len_new
            self.model = MLP(seq_len, pred_len, enc_in, self.period_len, mode='mean').float()
            self.model_std = MLP( seq_len, pred_len, enc_in, self.period_len, mode='std').float()
        
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
        
    def loss(self, true):
        # freq normalization
        B , O, N= true.shape
        residual, pred_main  = main_freq_part(true, self.freq_topk, self.rfft)
        

        lf = nn.functional.mse_loss
        
        if not self.SAN:
            return  lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual) 
        mean = self.preds_mean
        std = self.preds_std
        sliced_true = residual.reshape(B, -1, 12, N)
        return  lf(mean, sliced_true.mean(2)) + lf(std, sliced_true.std(2)) + lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual) 
        
        
    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        # freq normalization
        # input = norm_input.reshape(bs, len, dim)
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
        # freq prediction
        # seq_last = x_filtered[:, -1: , :].detach()
        # self.pred_main_freq_signal = self.model_freq((x_filtered - seq_last).transpose(1,2)).transpose(1,2) + seq_last # B O N
        # self.pred_main_freq_signal = self.model_freq((x_filtered).transpose(1,2)).transpose(1,2)  # B O N
        
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1,2), input.transpose(1,2)).transpose(1,2)
        
        if not self.SAN:
            return norm_input.reshape(bs, len, dim)
        
        input = norm_input
        # x_norm = norm_input

        # trend normalize
        input = input.reshape(bs, -1, self.period_len, dim)
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


    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        if self.SAN:
            # trend denormalize
            input = input_norm.reshape(bs, -1, self.period_len, dim)
            output = input * (self.preds_std.unsqueeze(2)+ self.epsilon) + self.preds_mean.unsqueeze(2)
            output = output.reshape(bs, len, dim)
            input_norm = output
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal
        
        return output.reshape(bs, len, dim)
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        
        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )
        
        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        
        self.tsmixer = TSMixer(256, enc_in, self.pred_len)


    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.tsmixer(self.model_all(inp).transpose(1,2)).transpose(1,2)
        
        
        

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