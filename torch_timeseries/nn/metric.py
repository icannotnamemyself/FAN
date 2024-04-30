from typing import Any
from sklearn.metrics import r2_score, classification_report

import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric
from torchmetrics import MetricCollection, R2Score, MeanSquaredError, SpearmanCorrCoef
from torchmetrics.functional.regression.r2 import _r2_score_compute, _r2_score_update
import numpy as np

def compute_r2(y_true, y_pred, aggr_mode='uniform_average'):
    if len(y_true.shape) <= 2:
        return r2_score(y_true, y_pred, multioutput=aggr_mode)
    
    if len(y_true.shape) == 3:
        # (batch, seq_len, n_nodes)
        # return np.mean([
        #     r2_score(y_true[i, ...], y_pred[i, ...], multioutput=aggr_mode) for i in range(y_true.shape[0])
        # ])
        return np.mean([
            r2_score(y_true[..., i], y_pred[..., i], multioutput=aggr_mode) for i in range(y_true.shape[-1])
        ])
    
    raise NotImplementedError(f'Cannot apply on y of {len(y_true.shape)} dims')

def _compute_corr_for_one_dim(y_true, y_pred):
    sigma_p = y_pred.std()
    if sigma_p == 0:
        return None
    
    sigma_g = y_true.std()
    mean_p = y_pred.mean()
    mean_g = y_true.mean()
    sigma_p += 1e-7
    sigma_g += 1e-7
    correlation = ((y_pred - mean_p) * (y_true - mean_g)).mean() / (sigma_p * sigma_g)
    return correlation

def compute_corr(y_true, y_pred):
    if (len(y_true.shape) == 2 and y_true.shape[-1] == 1) or (len(y_true.shape) == 1):
        return _compute_corr_for_one_dim(y_true, y_pred)
    
    # y size: (batch, seq_len, n_nodes)
    if len(y_true.shape) == 3:
        return np.mean([
            compute_corr(y_true[i, ...], y_pred[i, ...]) for i in range(y_true.shape[0])
        ])

    # y size: (n_samples, n_nodes)
    if len(y_true.shape) == 2:
        if isinstance(y_pred, torch.Tensor):
            sigma_p = y_pred.std(1, correction=0)  # (n_samples,)
        else:
            sigma_p = y_pred.std(1)  # (n_samples,)
            
        if isinstance(y_true, torch.Tensor):
            sigma_g = y_true.std(1, correction=0)  # (n_samples,)
        else:
            sigma_g = y_true.std(1)
        mean_p = y_pred.mean(1).reshape(-1, 1)  # (n_samples, 1)
        mean_g = y_true.mean(1).reshape(-1, 1)
        index = (sigma_g != 0)
        sigma_p += 1e-7
        sigma_g += 1e-7
        correlation = ((y_pred - mean_p) * (y_true - mean_g)).mean(1) / (sigma_p * sigma_g)
        correlation = correlation[index].mean().item()
        return correlation
    
    raise NotImplementedError(f'Cannot apply on y of {len(y_true.shape)} dims')

class TrendAcc(Metric):
    """Metric for single step forcasting"""
    compute_by_all : bool = False
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_trend_hit", default=tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, xt: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        assert len(preds.shape) == 2
        batch_size = preds.shape[0]
        self.sum_trend_hit += ((preds - xt) * (target - xt) > 0).sum()
        self.total += batch_size * preds.shape[1]

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return self.sum_trend_hit / self.total



class R2(R2Score):
    """Metric for multi step forcasting"""
    compute_by_all : bool = False

    def __init__(self, num_outputs, adjusted: int = 0, multioutput: str = "uniform_average") -> None:
        super().__init__(num_outputs, adjusted, multioutput)
        # if num_nodes > 1:
        #     self.add_state("sum_squared_error", default=torch.zeros(
        #         self.num_outputs, num_nodes), dist_reduce_fx="sum")
        #     self.add_state("sum_error", default=torch.zeros(
        #         self.num_outputs, num_nodes), dist_reduce_fx="sum")
        #     self.add_state("residual", default=torch.zeros(
        #         self.num_outputs, num_nodes), dist_reduce_fx="sum")
        #     self.add_state("total", default=torch.zeros(
        #         num_nodes), dist_reduce_fx="sum")

        # self.num_nodes = num_nodes

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        # [batch , seq, node] or (batch, node)
        
        # if len(preds.shape) > 2:
        #     bs, t, n = target.shape
        #     preds = preds.reshape(-1, self.num_outputs)
        #     target = target.reshape(-1, self.num_outputs)

        # if self.num_nodes > 1:
        #     sum_obs = torch.sum(target, dim=0)  # (seq, node) or (node)
        #     sum_squared_obs = torch.sum(
        #         target * target, dim=0)  # (seq, node)  or (node)
        #     residual = target - preds  # (batch , seq, node) or (node)
        #     rss = torch.sum(residual * residual, dim=0)  # (seq, node) or (node)
        #     n_obs = target.size(0)  # or (node)

        #     self.sum_squared_error += sum_squared_obs
        #     self.sum_error += sum_obs
        #     self.residual += rss
        #     self.total += n_obs
        # else:
        R2Score.update(self, preds, target)
    def compute(self) -> Tensor:
        # if self.num_nodes > 1:
        #     result = torch.tensor([_r2_score_compute(
        #         self.sum_squared_error[:, i], self.sum_error[:, i], self.residual[:,
        #                                                                             i], self.total[i], self.adjusted, self.multioutput
        #     ) for i in range(self.num_nodes)], device=self.device).mean()
        #     return result
        # else:
        return R2Score.compute(self)

class Corr(Metric):
    """correlation for multivariate timeseries, Corr compute correlation for every columns/nodes and output the averaged result"""
    compute_by_all : bool = True
    def __init__(self, save_on_gpu=False):
        super().__init__()
        self.save_on_gpu = save_on_gpu
        if save_on_gpu == True:
            self.add_state("y_pred", default=torch.Tensor(), dist_reduce_fx="cat")
            self.add_state("y_true", default=torch.Tensor(), dist_reduce_fx="cat")
        else:
            self.add_state("y_pred", default=[])
            self.add_state("y_true", default=[])


    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # if len(y_pred.shape) > 2:
        #     bs = y_pred.shape[0]
        #     y_pred = y_pred.reshape(bs, -1)
        #     y_true = y_true.reshape(bs, -1)
            
        if self.save_on_gpu == True:
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)
            self.y_true = torch.cat([self.y_true, y_true], dim=0)
        else:
            self.y_pred.append(y_pred.detach().cpu().numpy())
            self.y_true.append(y_true.detach().cpu().numpy())

        
    def compute(self):
        if self.save_on_gpu == True:
            return compute_corr(self.y_pred, self.y_true)
        else:
            return compute_corr(np.concatenate(self.y_pred, axis=0), np.concatenate(self.y_true, axis=0))


class RMSE(MeanSquaredError):
    def compute(self):
        return torch.sqrt(super().compute())