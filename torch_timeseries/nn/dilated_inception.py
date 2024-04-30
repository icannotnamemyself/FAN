import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F



class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2, kernel_set=[2,3,6,7]):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        assert cout > 0
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):

        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x
    
    
    # TODO: 输出 膨胀卷积结果长度
    def output_len():
        pass