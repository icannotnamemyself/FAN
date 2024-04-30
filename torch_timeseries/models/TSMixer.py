import torch.nn as nn
import torch.nn.functional as F

class TSLinear(nn.Module):
    def __init__(self, L, T):
        super(TSLinear, self).__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):
        return self.fc(x)

class TSMixer(nn.Module):
    def __init__(self, L, C, T, n_mixer:int = 8, dropout=0.05):
        # L: seq length
        # C: number of nodes
        # T:  output size 
        super(TSMixer, self).__init__()
        self.mixer_layers = []
        self.n_mixer = n_mixer
        for i in range(self.n_mixer):
            self.mixer_layers.append(MixerLayer(L, C, dropout))
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.temp_proj = TemporalProj(L, T)

    def forward(self, x):
        for i in range(self.n_mixer):
            x = self.mixer_layers[i](x)
        x = self.temp_proj(x)
        return x

class MLPTime(nn.Module):
    def __init__(self, T, dropout_rate=0.1):
        super(MLPTime, self).__init__()
        self.fc = nn.Linear(T, T)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MLPFeat(nn.Module):
    def __init__(self, C, dropout_rate=0.1):
        super(MLPFeat, self).__init__()
        self.fc1 = nn.Linear(C, C)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(C, C)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, L, C, dropout):
        super(MixerLayer, self).__init__()
        self.mlp_time = MLPTime(L, dropout)
        self.mlp_feat = MLPFeat(C, dropout)
        
    def batch_norm_2d(self, x):
        """ x has shape (B, L, C) """
        return (x - x.mean()) / x.std()
    
    def forward(self, x):
        """ x has shape (B, L, C) """
        res_x = x
        x = self.batch_norm_2d(x)
        x = x.transpose(1, 2)
        x = self.mlp_time(x)
        x = x.transpose(1, 2) + res_x
        res_x = x
        x = self.batch_norm_2d(x)
        x = self.mlp_feat(x) + res_x
        return x

class TemporalProj(nn.Module):
    def __init__(self, L, T):
        super(TemporalProj, self).__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x