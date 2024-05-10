import fire
import torch
from torch_timeseries.norm_experiments.experiment import NormExperiment
from torch_timeseries.models import TimesNet

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class TimesNetExperiment(NormExperiment):
    model_type: str = "TimesNet"
    
    label_len: int = 48
    d_model: int = 128
    e_layers: int = 2
    d_ff: int = 128  # out of memoery with d_ff = 2048
    num_kernels: int = 3
    top_k: int = 5
    dropout: float = 0.0
    embed: str = "timeF"
    freq: str = 'h'
    
    def _init_f_model(self):
        
        self.label_len = int(self.windows / 2)
        
        self.f_model = TimesNet(
            seq_len=self.windows, 
            label_len=self.label_len,
            pred_len=self.pred_len, 
            e_layers=self.e_layers, 
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            top_k=self.top_k,
            d_model=self.d_model,
            embed=self.embed,
            enc_in=self.dataset.num_features,
            freq=self.freq,
            dropout=self.dropout,
            c_out=self.dataset.num_features,
            task_name="long_term_forecast",
            )
        self.f_model = self.f_model.to(self.device)

    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, Steps,T)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
            
        # outputs:
            # pred: (B, O, N)
            # label:  (B,O,N)
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len :, :], batch_y_date_enc], dim=1
        )
        
        
        batch_x , dec_inp = self.model.normalize(batch_x, dec_inp=dec_inp) # (B, T, N)   # (B,L,N)
        
        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
        
        pred = self.model.denormalize(pred)

        return pred, batch_y # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred

def cli():
    fire.Fire(TimesNetExperiment)

if __name__ == "__main__":
    cli()

