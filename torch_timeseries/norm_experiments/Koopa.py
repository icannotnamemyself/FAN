import fire
import torch
from torch_timeseries.norm_experiments.experiment import NormExperiment
from torch_timeseries.models import Koopa

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class KoopaExperiment(NormExperiment):
    model_type: str = "Koopa"
    # label_len : int = 48
    
    alpha:float = 0.2


    seg_len : int = 48
    num_blocks : int = 3
    dynamic_dim : int = 128
    hidden_dim : int = 64
    hidden_layers : int = 2

    
    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        amps = 0.0
        
        for i, (
            batch_x,
            batch_y,
            origin_y,
            batch_x_date_enc,
            batch_y_date_enc,
        ) in enumerate(self.train_loader):
        # for data in self.train_loader:
            lookback_window = batch_x
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0]*self.alpha)).indices
        return mask_spectrum # as the spectrums of time-invariant component


    def _init_f_model(self):
        
        self.label_len = int(self.windows / 2)
        
        mask_spectrum = self._get_mask_spectrum()
        
        self.f_model = Koopa(
            mask_spectrum,
            self.dataset.num_features,
            self.windows,
            self.pred_len,
            self.seg_len,
            self.num_blocks,
            self.dynamic_dim,
            self.hidden_dim,
            self.hidden_layers,
            True,
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
    fire.Fire(KoopaExperiment)

if __name__ == "__main__":
    cli()
