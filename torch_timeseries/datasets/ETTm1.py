import os
import resource
from.dataset import Dataset, Freq, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data



class ETTm1(TimeSeriesDataset):
    name:str= 'ETTm1'
    num_features: int = 7
    freq : Freq = 't'
    length : int  = 69680
    windows : int = 384
    
    def download(self):
        download_url(
         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
         self.dir,
         filename='ETTm1.csv',
         md5="82d6bd89109c63d075d99c1077b33f38"
        )

        
    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'ETTm1.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    