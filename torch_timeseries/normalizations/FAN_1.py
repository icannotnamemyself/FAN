


import time
import numpy as np
import torch
import torch.nn as nn

    
def half_norm(input):
    # (B, T, N)
    bs, len, dim = input.shape
    M = len/2
    # x_sw = input.unfold(dimension=1, size=M, step=1) # (B, M, T-M+1, N)
    
    
    x_sw = input.unfold(dimension=1, size=int(M), step=1) # (B , T - T/2 + 1, N, T/2)

    patch_fft = torch.fft.rfft(x_sw, dim=3) # patched fft # (B , T - T/2 + 1, N, rfft_length)

    patch_fft_mean = patch_fft.abs().mean( dim=1, keepdim=True) # (B , 1, N, rfft_length)
    raylei_sigma = patch_fft_mean * np.sqrt(2/torch.pi) # (B , 1, N, rfft_length)

    patch_fft_norm = patch_fft/(raylei_sigma  + 1e-10) # (B , T - M + 1, N, rfft_length)

    norm_input_sw = torch.fft.irfft(patch_fft_norm, dim=3).real # (B , T - M + 1, N, rfft_length)
    norm_input = torch.concat([norm_input_sw[:, 0, :, :], norm_input_sw[:, -1, :, :]], dim=2) # (B,2, N, T/2)
    
    return norm_input.transpose(1,2), raylei_sigma
    

# def half_norm(input):
#     bs, len, dim = input.shape
#     M = len/2
#     x_sw = input.unfold(dimension=1, size=int(M), step=1) # (B , T - T/2 + 1, N, T/2)
#         # fft_result = torch.fft.rfft(x_sw, dim=2)

    
    
#     patch_fft = torch.fft.rfft(x_sw, dim=3) # patched fft # (B , T - T/2 + 1, N, T/2)

#     patch_fft_mean = patch_fft.abs().mean( dim=1, keepdim=True) # (B , 1, N, T/2)
#     raylei_sigma = patch_fft_mean * np.sqrt(2/torch.pi) # (B , 1, N, T/2)

#     patch_fft_norm = patch_fft/(raylei_sigma  + 1e-10) # (B , T - M + 1, N, T/2)

#     # sampled_fft = torch.concat([patch_fft_norm[:, 0, :, :], patch_fft_norm[:, -1, :, :]], dim=1)

#     estimate_raylei_sigma = raylei_sigma.squeeze(1) # (B, 1, N, T/2)
    

#     norm_input_sw = torch.fft.ifft(patch_fft_norm, dim=3).real # (B , T - M + 1, N, T/2)
#     norm_input = torch.concat([norm_input_sw[:, 0, :, :], norm_input_sw[:, -1, :, :]], dim=2) # (B, N, T)
    
#     return norm_input, estimate_raylei_sigma





class FAN(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        
        self.rfft_length = 1 + int(int(self.seq_len/2)/2)
        
        
        self.mlp = MLPfreq(self.rfft_length, seq_len, pred_len, enc_in)
        
    def loss(self, true):
        # freq normalization
        B , O, N= true.shape
        lf = nn.functional.mse_loss
        
        # _, estimate_raylei_sigma = half_norm(true)
        # print(1)
        return 0 #lf(self.pred_sigma , estimate_raylei_sigma.squeeze(1))
        
        
    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        M = len/2
        # x_sw = input.unfold(dimension=1, size=M, step=1) # (B, M, T-M+1, N)
        self.mean = input.mean(1, keepdim=True)
        
        input = input - self.mean

        x_sw = input.unfold(dimension=1, size=int(M), step=1) # (B , T - T/2 + 1, N, T/2)

        patch_fft = torch.fft.rfft(x_sw, dim=3) # patched fft # (B , T - T/2 + 1, N, rfft_length)

        patch_fft_mean = patch_fft.abs().mean( dim=1, keepdim=True) # (B , 1, N, rfft_length)
        raylei_sigma = patch_fft_mean * np.sqrt(2/torch.pi) # (B , 1, N, rfft_length)

        patch_fft_norm = patch_fft/(raylei_sigma  + 1e-10) # (B , T - M + 1, N, rfft_length)

        # sampled_fft = torch.concat([patch_fft_norm[:, 0, :, :], patch_fft_norm[:, -1, :, :]], dim=1)

        self.estimate_raylei_sigma = raylei_sigma.squeeze(1) # (B, 1, N, rfft_length)
        
        self.pred_sigma = self.mlp(self.estimate_raylei_sigma, input.transpose(1,2)) # (B, N, O/2)
 
        norm_input_sw = torch.fft.irfft(patch_fft_norm, dim=3).real # (B , T - M + 1, N, rfft_length)
        norm_input = torch.concat([norm_input_sw[:, 0, :, :], norm_input_sw[:, -1, :, :]], dim=2) # (B,2, N, T/2)
        
        return norm_input.transpose(1,2)


    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        M = int(len/2)
        
        # import pdb;pdb.set_trace()
        # x_sw = input_norm.reshape(bs, M, 2, dim)
        
        
        x_sw = input_norm.reshape(bs, -1, M, dim).permute(0, 1, 3, 2) # (B, 2, 2/T, N) PERMUTE -> b 2 n t/2
        patch_fft = torch.fft.rfft(x_sw, dim=3) * self.pred_sigma.unsqueeze(1)# patched fft # (B , 2, N, T/2)
        output = torch.fft.irfft(patch_fft, dim=3).real.permute(0, 1, 3, 2)# (B, 2, N, T/2) PERMUTE -> b T/2 2 N

        return output.reshape(bs, len, dim) + self.mean
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)


        
class MLPfreq(nn.Module):
    def __init__(self, sigma_len, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.sigma_len = sigma_len
        self.rfft_length = 1 + int(int(pred_len/2)/2)
        self.model_freq = nn.Sequential(
            nn.Linear(sigma_len, 64),
            nn.ReLU(),
        )
        
        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, self.rfft_length)
        )

        self.model = nn.Sequential(
            nn.Linear(sigma_len, 64),
            nn.ReLU(),
            nn.Linear(64, self.rfft_length),
            
        )


    def forward(self, x_sigma, x):
        # return self.model(x_sigma)
        
        
        inp = torch.concat([self.model_freq(x_sigma), x], dim=-1)
        return self.model_all(inp)

        
        





# class FAN1(nn.Module):
#     def __init__(self,  seq_len, pred_len, enc_in):
#         super().__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.enc_in = enc_in 
#         self.epsilon = 1e-8
        
#         self.rfft_length = 1 + int(int(self.seq_len/2)/2)
        
        
#         self.mlp = MLPfreq(self.rfft_length, seq_len, pred_len, enc_in)
        
#     def loss(self, true):
#         # freq normalization
#         B , O, N= true.shape
#         lf = nn.functional.mse_loss
        
#         # _, estimate_raylei_sigma = half_norm(true)
#         # print(1)
#         return 0 #lf(self.pred_sigma , estimate_raylei_sigma)
        
        
#     def normalize(self, input):
#         # (B, T, N)
#         bs, len, dim = input.shape
#         M = len/2
#         # x_sw = input.unfold(dimension=1, size=M, step=1) # (B, M, T-M+1, N)
        
        
#         x_sw = input.unfold(dimension=1, size=int(M), step=1) # (B , T - T/2 + 1, N, T/2)

#         patch_fft = torch.fft.rfft(x_sw, dim=3) # patched fft # (B , T - T/2 + 1, N, rfft_length)

#         patch_fft_mean = patch_fft.abs().mean( dim=1, keepdim=True) # (B , 1, N, rfft_length)
#         raylei_sigma = patch_fft_mean * np.sqrt(2/torch.pi) # (B , 1, N, rfft_length)

#         patch_fft_norm = patch_fft/(raylei_sigma.detach()  + 1e-10) # (B , T - M + 1, N, rfft_length)

#         # sampled_fft = torch.concat([patch_fft_norm[:, 0, :, :], patch_fft_norm[:, -1, :, :]], dim=1)

#         self.estimate_raylei_sigma = raylei_sigma.squeeze(1).detach() # (B, 1, N, rfft_length)
        
#         self.pred_sigma = self.mlp(self.estimate_raylei_sigma, input.transpose(1,2)) # (B, N, O/2)
 
#         norm_input_sw = torch.fft.irfft(patch_fft_norm, dim=3).real # (B , T - M + 1, N, rfft_length)
#         norm_input = torch.concat([norm_input_sw[:, 0, :, :], norm_input_sw[:, -1, :, :]], dim=2) # (B,2, N, T/2)
        
#         return norm_input.reshape(bs, len, dim)


#     def denormalize(self, input_norm):
#         # input:  (B, O, N)
#         # station_pred: outputs of normalize
#         bs, len, dim = input_norm.shape
#         M = int(len/2)
#         # input_fft = torch.fft.fft(input_norm, dim=1)
        
#         # x_sw = input_norm.unfold(dimension=1, size=int(M), step=1) # (B , T - T/2 + 1, N, T/2)
#         # patch_fft = torch.fft.rfft(x_sw, dim=3) # patched fft # (B , T - T/2 + 1, N, T/2)
        
#         # output = torch.fft.irfft(patch_fft * self.pred_sigma.unsqueeze(1), dim=3).real
#         # output = torch.concat([output[:, 0, :, :], output[:, -1, :, :]], dim=2) # (B, N, O)

        
#         # import pdb;pdb.set_trace()
#         x_sw = input_norm.reshape(bs, M, 2, dim).permute(0, 2, 3, 1) # (B, T/2, 2, N) PERMUTE -> b 2 n t/2
#         patch_fft = torch.fft.rfft(x_sw, dim=3) * self.estimate_raylei_sigma.detach().unsqueeze(1)# patched fft # (B , T - T/2 + 1, N, T/2)
#         output = torch.fft.irfft(patch_fft, dim=3).real.permute(0, 3, 1, 2)
#         return output.reshape(bs, len, dim)
    
#     def forward(self, batch_x, mode='n'):
#         if mode == 'n':
#             return self.normalize(batch_x)
#         elif mode =='d':
#             return self.denormalize(batch_x)


# class MLPfreq(nn.Module):
#     def __init__(self, sigma_len, seq_len, pred_len, enc_in):
#         super(MLPfreq, self).__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.channels = enc_in
#         self.sigma_len = sigma_len
#         self.rfft_length = 1 + int(int(pred_len/2)/2)
#         self.model_freq = nn.Sequential(
#             nn.Linear(sigma_len, 64),
#             nn.ReLU(),
#         )
        
#         self.model_all = nn.Sequential(
#             nn.Linear(64 + seq_len, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.rfft_length)
#         )


#     def forward(self, x_sigma, x):
#         inp = torch.concat([self.model_freq(x_sigma), x], dim=-1)
#         return self.model_all(inp)
        
        
        
