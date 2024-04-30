


import torch
import torch.nn as nn

class SampleConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super(SampleConv, self).__init__(in_channels, out_channels, kernel_size=(1,1))
        self.reset_parameters()  # 自定义的初始化方法

    def reset_parameters(self):
        # 初始化权重，均值为1，标准差为1
        nn.init.normal_(self.weight, mean=1, std=0.05)
        nn.init.normal_(self.bias, mean=0, std=0.1)
        # nn.init.constant_(self.conv.bias, 0)


class SNv4(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in, sample_nums=128, element_wise_affine=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.sample_nums = sample_nums
        self.epsilon = 1e-8
        self.element_wise_affine =element_wise_affine
        self._build_model()
            
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        if self.element_wise_affine:
            self.sample_parameter = nn.Parameter(1 + torch.randn(self.sample_nums, self.seq_len, self.enc_in))
        else:
            # self.upsample_layer = nn.Conv2d(1, self.sample_nums,(1,1))
            self.upsample_layer = SampleConv(1, self.sample_nums)
        self.z_model = MLP(self.seq_len, self.pred_len, self.enc_in)
        
        # self.model = MLP(self.seq_len, self.pred_len, self.enc_in, mode='mean')
        # self.model_std = MLP( self.seq_len, self.pred_len, self.enc_in, mode='std')

    def normalize(self, batch_x):
        # (B, T, N)

        if self.element_wise_affine:
            Q = batch_x.unsqueeze(-1).permute(0, 3,1,2) * self.sample_parameter.unsqueeze(0)
        else:
            Q = self.upsample_layer(batch_x.unsqueeze(-1).permute(0, 3,1,2)) # (B, C, T, N)

        mu = Q.mean(dim=1) # (B, T, N)
        std = Q.std(dim=1)

        input_mu = batch_x.mean(1, keepdim=True) # (B, 1, N)

        x_norm = (batch_x - mu) / std # # (B, T, N)
        
        
        # Z = self.z_model(Q - input_mu.unsqueeze(2)) +  mu[:, -1, :].unsqueeze(2) # (B C O N)
        # Z = self.weight[0] * self.z_model(Q - mu.unsqueeze(1)) +  self.weight[1] * input_mu[:, -1:, :].unsqueeze(2) # (B C O N)
        Z = self.z_model(Q - mu.unsqueeze(1)) +  input_mu.unsqueeze(2) # (B C O N)
        
        self.preds_mean = Z.mean(dim=1) # (B, O, N)
        self.preds_std = Z.std(dim=1) # (B, O, N)
        return x_norm


    def denormalize(self, input):
        # input:  (B, O, N)
        # station_pred: outputs of normalize 
        bs, len, dim = input.shape
        output = input*(self.preds_std) + self.preds_mean
        return output.reshape(bs, len, dim)
    
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)

class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.input = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_len),
        )
        
        
    def forward(self, x, x_norm=None):
        x = x.permute(0,1, 3, 2) # (B C N T)
        x = self.input(x)  # (B C N O)
        return x.permute(0, 1,3,2) # (B C O N)




# import torch
# import torch.nn as nn


# class SNv1(nn.Module):
#     def __init__(self,  seq_len, pred_len, enc_in, sample_nums=8, element_wise_affine=False):
#         super().__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.enc_in = enc_in 
#         self.sample_nums = sample_nums
#         self.epsilon = 1e-5
#         self._build_model()
#         self.weight = nn.Parameter(torch.ones(2, self.enc_in))

#     def _build_model(self):
#         self.upsample_layer = nn.Conv2d(1, self.sample_nums,(1,1))
#         self.z_model = MLP(self.seq_len, self.pred_len, self.enc_in)

#     def normalize(self, batch_x):
#         # (B, T, N)
#         Q = self.upsample_layer(batch_x.unsqueeze(-1).permute(0, 3,1,2)) # (B, C, T, N)
#         mu = Q.mean(dim=1) # (B, 1, T, N)
#         std = Q.std(dim=1)

#         input_mu = batch_x.mean(1, keepdim=True) # (B, 1, N)

#         x_norm = (batch_x - mu) / std
        
        
#         Z = self.z_model(Q - input_mu.unsqueeze(2)) + input_mu.unsqueeze(2) # (B C O N)
        
#         self.preds_mean = Z.mean(dim=1) # (B, O, N)
#         self.preds_std = Z.mean(dim=1) # (B, O, N)
#         return x_norm


#     def denormalize(self, input):
#         # input:  (B, O, N)
#         # station_pred: outputs of normalize 
#         bs, len, dim = input.shape
#         output = input*(self.preds_std) + self.preds_mean
#         return output.reshape(bs, len, dim)
    
    
#     def forward(self, batch_x, mode='n'):
#         if mode == 'n':
#             return self.normalize(batch_x)
#         elif mode =='d':
#             return self.denormalize(batch_x)

# class MLP(nn.Module):
#     def __init__(self, seq_len, pred_len, enc_in):
#         super(MLP, self).__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.channels = enc_in
#         self.input = nn.Sequential(
#             nn.Linear(self.seq_len, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.pred_len),
#         )
        
        
#     def forward(self, x):
#         x = x.permute(0,1, 3, 2) # (B C N T)
#         x = self.input(x)  # (B C N O)
#         return x.permute(0, 1,3,2) # (B C O N)

