
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_timeseries.norm_experiments.experiment import NormExperiment
import torch.nn as nn
from dataclasses import asdict,dataclass

from torch_timeseries.normalizations.No import No


class FFN(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, device, individual=False):
        super(FFN, self).__init__()
        
        hidden_dim = 16
        self.num_features = num_features
        self.linears = nn.ModuleList()
        
        for i in range(num_features):
            self.linears.append(
                nn.Sequential(
                    nn.Linear(seq_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, pred_len),
                )
            )
    
    def forward(self, x):
        # batch_x: (B, N, T)
        # out: (B, N, O)
        xs = []
        for i in range(self.num_features):
            xs.append(self.linears[i](x[:, i, :])) # x[:, i, :] =  
        x = torch.stack(xs, dim=1) # (B, N, O)
        
        return x
@dataclass
class ModelExperiment(NormExperiment):
    def _init_f_model(self):
        self.f_model = FFN(
            seq_len=self.windows,
            pred_len=self.pred_len,
            num_features=self.dataset.num_features,
            device=self.device,
        ).to(self.device)


    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, Steps,T)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
            
        # outputs:
            # pred: (B, O, N)
            # label:  (B,O,N)
        batch_x , _ = self.model.normalize(batch_x) # (B, T, N)   # (B,L,N)
        
        pred = self.model.fm(batch_x.transpose(1,2)).transpose(1,2) # (B, O, N)
        
        pred = self.model.denormalize(pred)

        return pred, batch_y # (B, O, N), (B, O, N)


def main():
    exp = ModelExperiment(
        dataset_type="DummyContinuous",
        data_path="./data",
        norm_type='RevIN', # No RevIN DishTS SAN 
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
    main()
