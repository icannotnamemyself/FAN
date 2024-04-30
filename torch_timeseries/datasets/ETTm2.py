import os
import resource
from.dataset import Dataset, Freq, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data



class ETTm2(TimeSeriesDataset):
    name:str= 'ETTm2'
    num_features: int = 7
    freq : Freq = 't'
    length : int  = 69680
    windows : int = 384
    
    def download(self):
        download_url(
         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
         self.dir,
         filename='ETTm2.csv',
         md5="7687e47825335860bf58bccb31be0c56"
        )

        
    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'ETTm2.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    
        