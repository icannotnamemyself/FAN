import torch
import torch.nn as nn
import torch.nn.functional as F
from class_resolver.contrib.torch import activation_resolver

class DilatedInception1d(nn.Module):
    kernel_set = [2, 3, 6, 7]
    max_kernel_size = max(kernel_set)

    def __init__(self, cin, cout, dilation_factor=2, dropout=0.3, act='relu'):
        super(DilatedInception1d, self).__init__()
        self.tconv = nn.ModuleList()
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv1d(cin, cout, kern, dilation=dilation_factor))
        self.dropout = nn.Dropout(dropout)
        self.act = activation_resolver.make(act)

    def forward(self, input):
        # input.shape: (B, C, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(-1):]
        x = torch.cat(x,dim=1)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DilatedInception2d(nn.Module):
    kernel_set = [2, 3, 6, 7]
    max_kernel_size = max(kernel_set)

    def __init__(self, cin, cout, dilation_factor=2, dropout=0.3, act='relu'):
        super(DilatedInception2d, self).__init__()
        self.tconv = nn.ModuleList()
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=dilation_factor))
        self.dropout = nn.Dropout(dropout)
        self.act = activation_resolver.make(act)

    def forward(self, input):
        # input.shape: (B, C, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(-1):]
        x = torch.cat(x,dim=1)
        x = self.act(x)
        x = self.dropout(x)
        return x
