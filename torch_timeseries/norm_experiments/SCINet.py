import fire
import torch
from torch_timeseries.norm_experiments.experiment import NormExperiment
from torch_timeseries.models.SCINet import SCINet

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field

from torch_timeseries.nn.metric import R2, Corr


@dataclass
class SCINetExperiment(NormExperiment):
    model_type: str = "SCINet"
    
    hid_size : int = 1
    num_stacks : int = 1
    num_levels : int = 3
    num_decoder_layer : int = 1
    concat_len : int = 0
    groups : int = 1
    kernel : int = 5
    dropout : float = 0.5
    single_step_output_One : int = 0
    input_len_seg : int = 1
    positionalE : bool = False
    modified : bool = True
    RIN : bool = False

    
    
    def _init_f_model(self):
        self.f_model = SCINet(
            output_len=self.pred_len, 
            input_len=self.windows, 
            input_dim = self.dataset.num_features,
            hid_size = self.hid_size, 
            num_stacks = self.num_stacks,
            num_levels = self.num_levels, 
            num_decoder_layer = self.num_decoder_layer, 
            concat_len = self.concat_len, 
            groups = self.groups, 
            kernel = self.kernel, 
            dropout = self.dropout,
            single_step_output_One = self.single_step_output_One, 
            input_len_seg = self.input_len_seg, 
            positionalE =self.positionalE, 
            modified = self.modified, 
            RIN=self.RIN
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
        
        batch_x , dec_inp = self.model.normalize(batch_x) # (B, T, N)   # (B,L,N)
        
        pred = self.model.fm(batch_x)
        
        pred = self.model.denormalize(pred)

        return pred, batch_y # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred

def cli():
    fire.Fire(SCINetExperiment)

def main():
    exp = SCINetExperiment(
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
    )

    exp.run()


if __name__ == "__main__":
    # main()
    cli()
