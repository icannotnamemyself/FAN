import fire
import torch
from torch_timeseries.norm_experiments.experiment import NormExperiment
from torch_timeseries.models import Informer

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class InformerExperiment(NormExperiment):
    model_type: str = "Informer"
    label_len: int = 48

    factor: int = 5
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layer: int = 512
    d_ff: int = 512
    # TODO: fix dropout to 0.3
    dropout: float = 0.0
    attn: str = "prob"
    embed: str = "fixed"
    activation = "gelu"
    distil: bool = True
    mix: bool = True

    def _init_f_model(self):
        self.f_model = Informer(
            self.dataset.num_features,
            self.dataset.num_features,
            self.dataset.num_features,
            self.pred_len,
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            activation=self.activation,
            distil=self.distil,
            mix=self.mix,
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
        dec_inp_label = batch_x[:, self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, self.label_len :, :], batch_y_date_enc], dim=1
        )
        
        
        batch_x , dec_inp = self.model.normalize(batch_x, dec_inp=dec_inp) # (B, T, N)   # (B,L,N)
        
        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
        
        pred = self.model.denormalize(pred)

        return pred, batch_y # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred

def cli():
    fire.Fire(InformerExperiment)

def main():
    exp = InformerExperiment(
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
        # lr=0.001,
        dropout=0.05,
        d_ff=256,
        # scaler_type="MaxAbsScaler",
    )

    exp.run()


if __name__ == "__main__":
    cli()
