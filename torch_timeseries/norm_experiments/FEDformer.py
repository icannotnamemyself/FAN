import fire
import torch
from torch_timeseries.norm_experiments.experiment import NormExperiment
from torch_timeseries.models import FEDformer

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class FEDformerExperiment(NormExperiment):
    model_type: str = "FEDformer"
    # label_len : int = 48
    version:str = 'Fourier'
    L : int = 3
    d_ff : int = 2048

    activation : str = 'gelu'
    e_layers : int = 2
    d_layers : int = 1
    mode_select : str = 'random'
    modes : int = 64
    output_attention : bool = True
    moving_avg : list = field(default_factory=lambda : [24])
    n_heads : int = 8
    cross_activation : str = 'tanh'
    d_model : int = 512
    embed : str = 'timeF'
    freq : str = 'h'
    dropout : float = 0.0
    base : str = 'legendre'
    
    l2_weight_decay : float = 0.0
    
    
    
    def _init_f_model(self):
        self.label_len = self.windows
        
        self.f_model = FEDformer(
        enc_in=self.dataset.num_features,
        dec_in=self.dataset.num_features,
        seq_len=self.windows,
        pred_len=self.pred_len,
        label_len=self.label_len, 
        c_out=self.dataset.num_features,  
        version=self.version, 
        L=self.L,
        d_ff=self.d_ff,
        activation=self.activation,
        e_layers=self.e_layers,
        d_layers=self.d_layers,
        mode_select=self.mode_select,
        modes=self.modes,
        output_attention=self.output_attention,
        moving_avg=self.moving_avg,
        n_heads=self.n_heads, 
        cross_activation=self.cross_activation, 
        d_model=self.d_model,
        embed=self.embed,
        freq=self.freq,
        dropout=self.dropout, 
        base=self.base
        
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
        
        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)[0]
        
        pred = self.model.denormalize(pred)

        return pred, batch_y # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred

def cli():
    fire.Fire(FEDformerExperiment)

def main():
    exp = FEDformerExperiment(
        dataset_type="ExchangeRate",
        data_path="./data",
        norm_type='RevIN', # No  DishTS
        optm_type="Adam",
        batch_size=128,
        device="cuda:1",
        windows=96,
        pred_len=96,
        horizon=1,
        epochs=100,
        dropout=0.05,
        d_ff=256,
    )

    exp.run()


if __name__ == "__main__":
    # main()
    cli()
