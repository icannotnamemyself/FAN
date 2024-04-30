
import torch
from turtle import forward
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_timeseries.datasets.electricity import Electricity
from torch_timeseries.nn.dilated_inception import DilatedInception
from torch_timeseries.nn.timeseries_startconv import TimeSeriesStartConv
import torch.nn.functional as F
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim import Optimizer , Adam
from torch.nn import init
import numbers
import torch.optim as optim

from torch_timeseries.data.scaler import MaxAbsScaler



class CNNDecoder(nn.Module):
    def __init__(self,  middle_channel=32, out_dim=3) -> None:
        super().__init__()
        self.end_conv_1 = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=middle_channel,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        
        
        
    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x