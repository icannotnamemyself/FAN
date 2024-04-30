


import torch
import torch.nn as nn


class SNv2(nn.Module):
    def __init__(self,  seq_len, pred_len, enc_in, sample_nums=8):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.sample_nums = sample_nums
        self.epsilon = 1e-8
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.sample_parameter = nn.Parameter(1 + 0.05 * torch.randn(self.sample_nums, self.seq_len, self.enc_in))
        self.model = MLP(self.seq_len, self.pred_len, self.enc_in, mode='mean')
        self.model_std = MLP( self.seq_len, self.pred_len, self.enc_in, mode='std')



    def normalize(self, batch_x):
        # (B, T, N)
        Q = batch_x.unsqueeze(-1).permute(0, 3,1,2) * self.sample_parameter.unsqueeze(0)
        mu = Q.mean(dim=1) # (B, 1, T, N)
        std = Q.std(dim=1)

        input_mu = batch_x.mean(1, keepdim=True) # (B, 1, N)

        x_norm = (batch_x - mu) / std
        
        # self.preds_mean = self.model( mu -  batch_x  , x_norm) * self.weight[0] + mu.mean(1, keepdim=True) * self.weight[1]
        self.preds_mean = self.model( mu -  input_mu[:, -1:, :]  , x_norm) * self.weight[0] + input_mu[:, -1:, :] * self.weight[1]
        # self.preds_mean = self.model( mu -  input_mu  , x_norm - input_mu) * self.weight[0] + input_mu * self.weight[1]
        
        self.preds_std = self.model_std(std, x_norm)
        return x_norm


    def denormalize(self, input):
        # input:  (B, O, N)
        # station_pred: outputs of normalize 
        bs, len, dim = input.shape
        output = input*(self.preds_std + self.epsilon) + self.preds_mean
        return output.reshape(bs, len, dim)
    
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)

class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, mode):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_hat = nn.Linear(self.seq_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_norm):
        x, x_norm = x.permute(0, 2, 1), x_norm.permute(0, 2, 1) # (B, N, T) 
        x = self.input(x)
        x_norm = self.input_hat(x_norm)
        x = torch.cat([x, x_norm], dim=-1) # (B, )
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1) # (B, O, N)



# import torch

# # 假设的输入尺寸和数据
# N, T, C = 5, 3, 4  # 仅作为示例
# x1 = torch.randn(N, T, C)  # 随机生成的示例数据
# x2 = torch.randn(N, T, C)  # 随机生成的示例数据

# # 初始化一个空的协方差矩阵
# cov_matrix = torch.zeros(N, N, T)

# # 计算协方差矩阵
# for t in range(T):
#     # 对于每个时间点，我们提取所有N个变量的C个样本
#     samples_x1 = x1[:, t, :]
#     samples_x2 = x2[:, t, :]
    
#     # 计算均值
#     mean_x1 = torch.mean(samples_x1, dim=1, keepdim=True)
#     mean_x2 = torch.mean(samples_x2, dim=1, keepdim=True)
    
#     # 中心化
#     samples_x1_centered = samples_x1 - mean_x1
#     samples_x2_centered = samples_x2 - mean_x2
    
#     # 计算协方差
#     for i in range(N):
#         for j in range(N):
#             cov_matrix[i, j, t] = torch.sum(samples_x1_centered[i] * samples_x2_centered[j]) / (C - 1)

# # cov_matrix.shape, cov_matrix



# import numpy as np

# # 假设的多维高斯分布参数
# mu = np.array([1, 2])  # 均值向量
# Sigma = np.array([[1, 0.5], [0.5, 2]])  # 协方差矩阵

# # 特征分解协方差矩阵
# eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

# # 构造D^{-1/2}
# D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))

# # 计算变换矩阵P^T * D^{-1/2}
# transform_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T

# # 示例数据点x (可以是多个点组成的矩阵)
# x = np.random.multivariate_normal(mu, Sigma, size=1000)  # 生成一些符合这个分布的数据点

# # 归一化过程
# # 去中心化
# x_centered = x - mu

# # 应用线性变换
# z = x_centered @ transform_matrix

# # 验证变换后的数据的协方差矩阵是否为单位矩阵
# cov_z = np.cov(z, rowvar=False)

# transform_matrix, cov_z
