from typing import Any
from sklearn.metrics import r2_score, classification_report

import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric
from torchmetrics import MetricCollection, R2Score, MeanSquaredError, SpearmanCorrCoef
import numpy as np


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



# def masked_mape(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels!=null_val)
#     mask = mask.float()
#     mask /=  torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds-labels)/labels
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)



class MaskedMAPE(Metric):
    """correlation for multivariate timeseries, Corr compute correlation for every columns/nodes and output the averaged result"""
    compute_by_all : bool = True
    def __init__(self, save_on_gpu=False,null_val=np.nan):
        super().__init__()
        self.save_on_gpu = save_on_gpu
        self.null_val = null_val
        if save_on_gpu == True:
            self.add_state("y_pred", default=torch.Tensor(), dist_reduce_fx="cat")
            self.add_state("y_true", default=torch.Tensor(), dist_reduce_fx="cat")
        else:
            self.add_state("y_pred", default=[])
            self.add_state("y_true", default=[])

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.save_on_gpu == True:
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)
            self.y_true = torch.cat([self.y_true, y_true], dim=0)
        else:
            self.y_pred.append(y_pred.detach().cpu())
            self.y_true.append(y_true.detach().cpu())

        
    def compute(self):
        if self.save_on_gpu == True:
            return masked_mape(self.y_pred,self.y_true,self.null_val)
        else:
            return masked_mape(torch.concat(self.y_pred, axis=0), torch.concat(self.y_true, axis=0),self.null_val)

