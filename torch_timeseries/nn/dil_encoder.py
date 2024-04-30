
import torch
from torch import Tensor
import torch.nn as nn
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

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    """Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # layer_norm_affline 会增大 参数个数。
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class DILEncoder(nn.Module):
    def __init__(self, input_node:int,seq_len:int,in_dim:int,  middle_channel=32,dilation_exponential=1, layers=3, dropout=0.3) -> None:
        super().__init__()
        
        self.input_node = input_node
        
        self.dropout = dropout
        
        self.start_conv = TimeSeriesStartConv(channel_in=in_dim, channel_out=middle_channel)
        self.filter_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.dil_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
    

        self.idx = torch.arange(self.input_node)
        self.seq_len = seq_len # 输入窗口长度 window
        self.layers = layers

        # 数据长度应该大于=感受野，否则kernelsize 需要发生变化
        max_kernel_size = 7
        if dilation_exponential>1:
            self.max_receptive_field = int(1+(max_kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1)) 
        else:
            self.max_receptive_field = layers*(max_kernel_size-1) + 1

        kernel_size = 7
        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                # self.dil_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))
                self.filter_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=middle_channel,
                                                out_channels=middle_channel,
                                                kernel_size=(1, 1)))

                layer_norm_affline = False
                if self.seq_len>self.max_receptive_field:
                    self.norm.append(LayerNorm((middle_channel, input_node, self.seq_len - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((middle_channel, input_node, self.max_receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
    def forward(self,input:Tensor):
        """_summary_
        Args:
            x (Tensor): shape (b ,aux_dim, n , p)
            b: batch_size
            aux_dim : normally 1
            n: node num
            p: window length
            
        Return: (b, n , m)
            b: batch_size
            n: node num
            m: steps
        """
        if self.seq_len<self.max_receptive_field:
            input = nn.functional.pad(input,(self.max_receptive_field-self.seq_len,0,0,0))
        
        
        x = self.start_conv(input)  # out: (b, embed_dim, n, p )
        
        for i in range(self.layers):
            residual = x
        
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            
            x = filter * gate # (n , 1 , m , p)
            x = F.dropout(x, self.dropout, training=self.training) # (n , 1 , m , p)
            
            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.norm[i](x,self.idx)
                
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        return x
        
        