import torch
import torch.nn as nn


class No(nn.Module):
    def __init__(self):
        super().__init__()
        
    def loss(self, x1):
        return 0
    def normalize(self, batch_x):
        return batch_x

    def denormalize(self, batch_x):
        return batch_x

