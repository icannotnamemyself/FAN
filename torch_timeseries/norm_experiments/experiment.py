import datetime
import json
import os
import random
import re
import signal
import threading
import time
import hashlib
from prettytable import PrettyTable
import sys
####
from typing import Dict, List, Type, Union
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm
import wandb
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.dataloader import (
    ChunkSequenceTimefeatureDataLoader,
)
from torch.nn import MSELoss, L1Loss

from torch.optim import Optimizer, Adam
from dataclasses import asdict, dataclass, field
from torch_timeseries.normalizations import *

from torch_timeseries.nn.metric import R2, Corr, TrendAcc,RMSE, compute_corr, compute_r2
from torch_timeseries.metrics.masked_mape import MaskedMAPE
from torch_timeseries.norm_experiments.Model import Model
from torch_timeseries.utils.early_stopping import EarlyStopping
import json
import codecs

# class Task(Enum):
#     SingleStepForecast : str = "single_step_forecast"
#     MultiStepForecast : str = "multi_steps_forecast"
#     Imputation : str = "imputation"
#     Classification : str = "classifation"
#     AbnomalyDetection : str = "abnormaly_detection"


def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2):
    k = mu1.size(1)
    
    Sigma2_inv = torch.linalg.inv(Sigma2)
    
    tr_term = torch.einsum('bij,bjk->bi', Sigma2_inv, Sigma1)
    
    mu_diff = mu2 - mu1
    mu_term = torch.einsum('bi,bij,bj->b', mu_diff, Sigma2_inv, mu_diff)
    
    det_term = torch.log(torch.linalg.det(Sigma2) / torch.linalg.det(Sigma1))
    
    kl_div = 0.5 * (tr_term + mu_term - k + det_term)
    
    return kl_div.sum()

@dataclass
class ResultRelatedSettings:
    dataset_type: str
    optm_type: str = "Adam"
    model_type: str = ""
    scaler_type: str = "StandarScaler"
    loss_func_type: str = "mse"
    batch_size: int = 32
    lr: float = 0.0003
    l2_weight_decay: float = 0.0005
    epochs: int = 100

    horizon: int = 3
    windows: int = 384
    pred_len: int = 1

    patience: int = 5
    max_grad_norm: float = 5.0
    invtrans_loss: bool = False
    
    norm_type : str = ''
    norm_config : dict = field(default_factory= lambda : {})
    

@dataclass
class Settings(ResultRelatedSettings):
    data_path: str = "./data"
    device: str = "cuda:0"
    num_worker: int = 20
    save_dir: str = "./results"
    experiment_label: str = str(int(time.time()))


def count_parameters(model, print_fun=print):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print_fun(table)
    print_fun(f"Total Trainable Params: {total_params}")
    return total_params


class NormExperiment(Settings):
    def config_wandb(
        self,
        project: str,
        name: str,
    ):

        # TODO: add seeds config parameters
        def convert_dict(dictionary):
            converted_dict = {}
            for key, value in dictionary.items():
                converted_dict[f"config.{key}"] = value
            return converted_dict

        # check wether this experiment had runned and reported on wandb
        api = wandb.Api()
        config_filter = convert_dict(self.result_related_config)
        runs = api.runs(path=project, filters=config_filter)
        
        try:
            if runs[0].state == "finished" or runs[0].state == "running":
                print(f"{self.model_type} {self.dataset_type} w{self.windows} w{self.horizon}  Experiment already reported, quiting...")
                self.finished = True
                return 
        except:
            pass
        
        
        run = wandb.init(
            mode='offline',
            project=project,
            name=name,
            tags=[self.model_type, self.dataset_type, f"horizon-{self.horizon}", f"window-{self.windows}", f"pred-{self.pred_len}", f"{self.norm_type}"],
        )
        wandb.config.update(asdict(self))
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self
    
    def wandb_sweep(
        self,
        project,
        name,
    ):
        run = wandb.init(
            project='BiSTGNN'
        )
        wandb.config.update(asdict(self))
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self

    
    def _use_wandb(self):
        return hasattr(self, "wandb")

    
    def _run_print(self, *args, **kwargs):
        time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
        
        print(*args, **kwargs)
        if hasattr(self, "run_setuped") and getattr(self, "run_setuped") is True:
            with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
                print(time, *args, flush=True, file=f)

    
    def config_wandb_verbose(
        self,
        project: str,
        name: str,
        tags: List[str],
        notes: str,
    ):
        run = wandb.init(
            project=project,
            name=name,
            notes=notes,
            tags=tags,
        )
        wandb.config.update(asdict(self))
        print(f"using wandb , running in config: {asdict(self)}")
        self.wandb = True
        return self

    def _init_loss_func(self):
        loss_func_map = {"mse": MSELoss, "l1": L1Loss}
        self.loss_func = loss_func_map[self.loss_func_type]()

    def _init_metrics(self):
        if self.pred_len == 1:
            self.metrics = MetricCollection(
                metrics={
                    "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                    "r2_weighted": R2(
                        self.dataset.num_features, multioutput="variance_weighted"
                    ),
                    "mse": MeanSquaredError(),
                    "corr": Corr(),
                    "mae": MeanAbsoluteError(),
                }
            )
        elif self.pred_len > 1:
            self.metrics = MetricCollection(
                metrics={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                    "mape": MaskedMAPE(null_val=0),
                    # "mape": MeanAbsolutePercentageError(),
                    'rmse': RMSE(),
                }
            )

        self.metrics.to(self.device)

    
    
    @property
    def result_related_config(self):
        ident = asdict(self)
        keys_to_remove = [
            "data_path",
            "device",
            "num_worker",
            "save_dir",
            "experiment_label",
        ]
        for key in keys_to_remove:
            if key in ident:
                del ident[key]
        return ident

    def _run_identifier(self, seed) -> str:
        ident = self.result_related_config
        ident["seed"] = seed
        # only influence the evluation result, not included here
        # ident['invtrans_loss'] = False

        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return str(ident_md5)

    def _init_data_loader(self):
        self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=False,
            shuffle_train=True,
            freq="h",
            batch_size=self.batch_size,
            train_ratio=0.7,
            val_ratio=0.2, # 0.1
            num_worker=self.num_worker,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")

    def _init_optimizer(self):
        self.model_optim = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optim, T_max=self.epochs
        )
        # if not isinstance(self.n_model, No):
        #     self.norm_model_optim = Adam(
        #         self.n_model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        #     )

    def _init_n_model(self):
        print(f"using {self.norm_type} as normalization model., config: {self.norm_config}")
        Ty = self._parse_type(self.norm_type) 
        if self.norm_type == 'RevIN':
            self.n_model : torch.nn.Module = Ty(self.dataset.num_features, True, **self.norm_config)
        elif self.norm_type == 'SAN':
            self.n_model : torch.nn.Module = Ty(self.windows, self.pred_len, 12, self.dataset.num_features, **self.norm_config)
        elif self.norm_type == 'DishTS':
            self.n_model : torch.nn.Module = Ty(self.dataset.num_features, self.windows, **self.norm_config)
        elif self.norm_type == 'CovidsV1':
            self.n_model : torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features, **self.norm_config)
        elif self.norm_type == 'SNv1':
            self.n_model : torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features, **self.norm_config)
        elif self.norm_type == 'No':
            self.n_model : torch.nn.Module = No(**self.norm_config)
        else:
            self.n_model : torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features, **self.norm_config)
        self.n_model = self.n_model.to(self.device)
        

    def _init_f_model(self) -> torch.nn.Module:
        self.f_model = None
        raise NotImplementedError()

        
    def _init_model(self):
        # self.model = self._parse_type(self.model_type)().to(self.device)
        self.model =   Model(self.model_type, self.f_model,self.n_model).to(self.device)



    def _setup(self):
        # init data loader
        self._init_data_loader()

        # init metrics
        self._init_metrics()

        # init loss function based on given loss func type
        self._init_loss_func()

        self.current_epochs = 0
        self.current_run = 0

        self.setuped = True

    def _setup_run(self, seed):
        # setup experiment  only once
        if not hasattr(self, "setuped"):
            self._setup()
        # setup torch and numpy random seed
        self.reproducible(seed)
        # init model, optimizer and loss function

        self._init_n_model()
        
        self._init_f_model()
        
        self._init_model()


        self._init_optimizer()
        self.current_epoch = 0
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            f"w{self.windows}h{self.horizon}s{self.pred_len}",
            self._run_identifier(seed),
        )

        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )

        self.run_checkpoint_filepath = os.path.join(
            self.run_save_dir, "run_checkpoint.pth"
        )

        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )
        self.run_setuped = True
        
        
        
    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")


    def _save(self, seed=0):
        self.checkpoint_path = os.path.join(
            self.save_dir, f"{self.model_type}/{self.dataset_type}"
        )
        self.checkpoint_filepath = os.path.join(
            self.checkpoint_path, f"{self.experiment_label}.pth"
        )
        # 检查目录是否存在
        if not os.path.exists(self.checkpoint_path):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.checkpoint_path)
            print(f"Directory '{self.checkpoint_path}' created successfully.")

        self.app_state = {
            "model": self.model,
            # "n_model": self.n_model,
            "optimizer": self.model_optim,
            # "norm_model_optim": self.norm_model_optim,
            
        }

        self.app_state.update(asdict(self))

        # now only save the newest state
        torch.save(self.app_state, f"{self.checkpoint_filepath}")

    def _save_run_check_point(self, seed):
        # 检查目录是否存在
        if not os.path.exists(self.run_save_dir):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

            
        self.run_state = {
            # "n_model": self.n_model.state_dict(),
            # "norm_model_optim": self.norm_model_optim.state_dict(),
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }
        # if not isinstance(self.n_model, No):
        #     self.run_state['n_model'] =  self.n_model.state_dict()
            # self.run_state['norm_model_optim'] =  self.norm_model_optim.state_dict()

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def reproducible(self, seed):
        # for reproducibility
        # torch.set_default_dtype(torch.float32)
        print("torch.get_default_dtype()", torch.get_default_dtype())
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True

    def _process_input(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, dec_inp=None, dec_input_date=None):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, Steps,T)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
        # outputs:
            # pred: (B, O, N)
        raise NotImplementedError()
        # batch_x = batch_x.transpose(1,2) # (B, N, T)
        # batch_x_date_enc = batch_x_date_enc.transpose(1,2) # (B, N, T)
        # pred = self.model(batch_x) # (B, O, N)
        # pred = pred.transpose(1,2) # (B, O, N)
        # return pred
    


    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, Steps,T)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
            
        # outputs:
            # pred: (B, O, N)
            # label:  (B,O,N)
        raise NotImplementedError()
        # label_len = 48
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # dec_inp_label = batch_x[:, label_len :, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, label_len :, :], batch_y_date_enc], dim=1
        # )
        
        # pred = self._process_input(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, dec_inp, dec_inp_date_enc)

        # return pred, batch_y # (B, O, N), (B, O, N)

    def _evaluate(self, dataloader):
        self.model.eval()
        # self.n_model.eval()
        self.metrics.reset()
        
        length = 0
        if dataloader is self.train_loader:
            length = self.dataloader.train_size
        elif dataloader is self.val_loader:
            length = self.dataloader.val_size
        elif dataloader is self.test_loader:
            length = self.dataloader.test_size

        # y_truths = []
        # y_preds = []
        with torch.no_grad():
            with tqdm(total=length,position=0, leave=True) as progress_bar:
                for batch_x, batch_y,batch_origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                    batch_size = batch_x.size(0)
                    batch_x = batch_x.to(self.device, dtype=torch.float32)
                    batch_y = batch_y.to(self.device, dtype=torch.float32)
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                    batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                    preds, truths = self._process_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    ) # preds : (B, O, N) truths: (B O N)
                    # the result should be the same
                    # self.metrics.update(preds.view(batch_size, -1), truths.view(batch_size, -1))
                    # import pdb;pdb.set_trace()
                    batch_origin_y = batch_origin_y.to(self.device)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch_origin_y

                    if self.pred_len == 1:
                        self.metrics.update(preds.view(batch_size, -1), truths.view(batch_size, -1))
                    else:
                        self.metrics.update(preds.contiguous(), truths.contiguous())

                    progress_bar.update(batch_x.shape[0])
                    
                    
                    # y_preds.append(preds)
                    # y_truths.append(truths)
                    
            # y_preds = torch.concat(y_preds, dim=0)
            # y_truths = torch.concat(y_truths, dim=0)

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result

    
    
    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)


        for name, metric_value in test_result.items():
            if self._use_wandb():
                wandb.run.summary["test_" + name] = metric_value
                # result = {}
                # for name,value in test_result.items():
                #     result['val_' + name] = value
                # wandb.log(result, step=self.current_epoch)

        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Evaluating .... ")
        val_result = self._evaluate(self.val_loader)
        
        
        for name, metric_value in val_result.items():
            if self._use_wandb():
                wandb.run.summary["val_" + name] = metric_value

        self._run_print(f"vali_results: {val_result}")
        return val_result

    def _train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps,position=0, leave=True) as progress_bar:
            self.model.train()
            # self.n_model.train()
            # import pdb;pdb.set_trace()
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device)
                self.model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                pred, true = self._process_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                
                # 
                if  isinstance(self.model.nm, SAN):
                    mean = self.model.pred_stats[:, :, :self.dataset.num_features]
                    std = self.model.pred_stats[:, :, self.dataset.num_features:]
                    sliced_true = true.reshape(bs, -1, 12, self.dataset.num_features)
                    loss = self.loss_func(pred, true) + self.loss_func(mean, sliced_true.mean(2)) + self.loss_func(std, sliced_true.std(2))
                # elif  isinstance(self.model.n_model, DishTS):
                    # mean = self.model.pred_stats[:, :, :self.dataset.num_features]
                    # std = self.model.pred_stats[:, :, self.dataset.num_features:]
                    # loss2 = self.loss_func(mean, true.mean(1)) + self.loss_func(std, true.std(1))
                    # loss2.backward()
                elif  isinstance(self.model.nm, PeriodV2):
                    mean = self.model.nm.preds_mean
                    std = self.model.nm.preds_std
                    sliced_true = true.reshape(bs, -1, 12, self.dataset.num_features)
                    loss = self.loss_func(pred, true) + self.loss_func(mean, sliced_true.mean(2)) + self.loss_func(std, sliced_true.std(2))
                    
                elif isinstance(self.model.nm, PeriodFDV4) or  isinstance(self.model.nm, PeriodFDV5) or isinstance(self.model.nm, FAN):
                    loss = self.loss_func(pred, true) + self.model.nm.loss(true)

                elif  isinstance(self.model.nm, PeriodFDV3):
                    mean = self.model.nm.preds_mean
                    std = self.model.nm.preds_std
                    sliced_true = true.reshape(bs, -1, 12, self.dataset.num_features)
                    loss = self.loss_func(pred, true) + self.loss_func(mean, sliced_true.mean(2)) + self.loss_func(std, sliced_true.std(2)) \
                         + self.model.nm.loss(true)


                elif  isinstance(self.model.nm, CovidsV2):
                    
                    period_len = self.model.nm.period_len
                    mean = self.model.nm.preds_mean
                    sigma = self.model.nm.preds_sigma
                    
                    sliced_true = true.reshape(bs, -1, period_len, self.dataset.num_features) # (B, M', T, N)
                    sliced_mean = torch.mean(sliced_true, dim=-2, keepdim=True) # (B, M, 1, N)
                    input_center = sliced_true - sliced_mean # (B, M, T, N)
                    sigma_true = torch.einsum('bmtv,bmtn->bmvn', input_center, input_center) / (period_len - 1) # (B, M, N, N)
                    
                    loss = self.loss_func(pred, true) + self.loss_func(mean, sliced_true.mean(2)) + self.loss_func(sigma , sigma_true)
                elif  isinstance(self.model.nm, SNv2):
                    mean = self.model.nm.preds_mean
                    std = self.model.nm.preds_std
                    loss = self.loss_func(pred, true) + self.loss_func(mean, true) + self.loss_func(std[:, -1, :], true.std(1))
                elif  isinstance(self.model.nm, SNv4):
                    mean = self.model.nm.preds_mean
                    std = self.model.nm.preds_std
                    loss = self.loss_func(pred, true) + self.loss_func(mean, true) 
                    

                elif  isinstance(self.model.nm, CovidsV3) or isinstance(self.model.nm, CovidsV4):
                    period_len = self.model.nm.period_len
                    N = self.dataset.num_features
                    sliced_true = true.reshape(bs, -1, period_len, N) # (B, M', T, N)
                    sliced_mean = torch.mean(sliced_true, dim=-2, keepdim=True) # (B, M, 1, N)
                    input_center = sliced_true - sliced_mean # (B, M, T, N)

                    sigma_true = torch.einsum('bmtv,bmtn->bmvn', input_center, input_center) / (period_len - 1) # (B, M, N, N)
                    
                    nm = self.model.nm
                    mean = self.model.nm.preds_mean
                    loss = self.loss_func(pred, true) \
                        + ((nm.x_norm_mean)**2).mean()  + ((nm.Sigma_norm - torch.eye(N).to(self.device))**2).mean() \
                        + ((nm.preds_mean - sliced_mean.squeeze(2))**2).mean()  + ((nm.output_Sigma - sigma_true)**2).mean()
                        # kl_divergence_gaussian(mean, self.model.nm., , ,sigma_true.reshape(-1, ))
                
                else:
                    loss = self.loss_func(pred, true)

                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

                self.model_optim.step()
            return train_loss
    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with codecs.open(os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists

    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")



        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)


        # if not isinstance(self.n_model, No):
        #     self.n_model.load_state_dict(check_point["n_model"])
        #     self.norm_model_optim.load_state_dict(check_point["norm_model_optim"])

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])
       
    def _resume_from(self, path):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(path, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

       
    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))


    def single_step_forecast(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self.experiment_label = f"{self.model_type}-w{self.windows}-h{self.horizon}"
    
    
    def run_more_epochs(self, seed=42, epoches=200) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self.epoches = epoches

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        
        self.model_parameters_num = self.count_parameters(self._run_print)
        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num
        # for resumable reproducibility

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                break

            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            # for resumable reproducibility
            self.reproducible(seed + self.current_epoch)
            
            train_losses =  self._train()
            
            

            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )
            self._run_print(
                f"Traininng loss : {np.mean(train_losses)}"
            )
            
            


            # self._run_print(f"Val on train....")
            # trian_val_result = self._evaluate(self.train_loader)
            # self._run_print(f"Val on train result: {trian_val_result}")
            
            # evaluate on val set
            val_result = self._val()

            # test
            test_result = self._test()

            if self._use_wandb():
                result = {'train_loss': float( np.mean(train_losses)) }
                for name,value in val_result.items():
                    result['val_' + name] = value
                for name,value in test_result.items():
                    result['test_' + name] = value
                wandb.log(result, step=self.current_epoch)


            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.loss_func_type], model=self.model)
            
            self._save_run_check_point(seed)

            self.scheduler.step()
            
            # if self._use_wandb():
            #     wandb.log(result, step=self.current_epoch)



        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False
        return best_test_result
        

    
    def test_sweep(self):
        run = wandb.init(
            mode='offline',
            project='sweep_test'
        )
        wandb.log({"log1":1})
        run.finish()
        
        # raise NotImplementedError("ss")
        return

    def count_parameters(self, print_fun):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print_fun(table)
        print_fun(f"Total Trainable Params: {total_params}")
        return total_params
    
    
    # def _norm(self, batch_x):
    #     if self.norm_type == "RevIN":
    #         batch_x, dec_inp = self.n_model.normalize(inputx)
    #         batch_x = self.n_model(batch_x)
    #     elif self.norm_type == "SAN":
    #         batch_x, pred_stats = self.n_model.normalize(inputx)
    #         batch_x = self.n_model(batch_x)
    #     elif self.norm_type == "DishTS":
    #         batch_x, dec_inp = self.n_model.normalize(inputx)

    
    def run(self, seed=42) -> Dict[str, float]:
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return {}

        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        
        self.model_parameters_num = self.count_parameters(self._run_print)
        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num
        # for resumable reproducibility

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                break

            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            # for resumable reproducibility
            self.reproducible(seed + self.current_epoch)
            train_losses =  self._train()

            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )
            self._run_print(
                f"Traininng loss : {np.mean(train_losses)}"
            )

            # self._run_print(f"Val on train....")
            # trian_val_result = self._evaluate(self.train_loader)
            # self._run_print(f"Val on train result: {trian_val_result}")
            
            # evaluate on val set
            result = self._val()
            # test
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(result[self.loss_func_type], model=self.model)
            
            self._save_run_check_point(seed)

            self.scheduler.step()
            
            # if self._use_wandb():
            #     wandb.log(result, step=self.current_epoch)



        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False
        return best_test_result

    def dp_run(self, seed=42, device_ids: List[int] = [0, 2, 4, 6], output_device=0):
        self._setup_dp_run(seed, device_ids, output_device)
        print(f"run : {self.current_run} in seed: {seed}")
        print(
            f"model parameters: {sum([p.nelement() for p in self.model.parameters()])}"
        )
        epoch_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            if self._use_wandb():
                wandb.run.summary["at_epoch"] = epoch
            self._train()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # evaluate on vali set
            self._val()

            self._save(seed=seed)

        return self._test()

    def runs(self, seeds: List[int] = [42,43,44]):
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return 
        if self._use_wandb():
            wandb.config.update({"seeds": seeds})

        
        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            if self._use_wandb():
                wandb.run.summary["at_run"] = i
            torch.cuda.empty_cache()
            result = self.run(seed=seed)
            torch.cuda.empty_cache()

            results.append(result)

            if self._use_wandb():
                for name, metric_value in result.items():
                    wandb.run.summary["test_" + name] = metric_value

        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
            )
        )
        if self._use_wandb():
            for index, row in self.metric_mean_std.iterrows():
                wandb.run.summary[f"{index}_mean"] = row["mean"]
                wandb.run.summary[f"{index}_std"] = row["std"]
                wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"
        wandb.finish()
        # return self.metric_mean_std

def main():
    exp = Experiment(
        dataset_type="ETTm1",
        data_path="./data",
        optm_type="Adam",
        model_type="Informer",
        batch_size=32,
        device="cuda:3",
        windows=10,
        epochs=1,
        lr=0.001,
        pred_len=3,
        scaler_type="MaxAbsScaler",
    )

    # exp = Experiment(settings)
    # exp.run()
# This function forcibly kills the remaining wandb process.
def force_finish_wandb():
    with open(os.path.join(os.path.dirname(__file__), './wandb/latest-run/logs/debug-internal.log'), 'r') as f:
        last_line = f.readlines()[-1]
    match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
    if match:
        pid = int(match.group(2))
        print(f'wandb pid: {pid}')
    else:
        print('Cannot find wandb process-id.')
        return
    
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError:
        print(f"Failed to kill process with PID {pid}.")

# Start wandb.finish() and execute force_finish_wandb() after 60 seconds.
def try_finish_wandb():
    threading.Timer(5, force_finish_wandb).start()
    wandb.finish()

# trainning scripts

# use try_finish_wandb instead of wandb.finish
# try_finish_wandb()

if __name__ == "__main__":
    main()
