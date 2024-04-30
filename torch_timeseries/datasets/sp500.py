import os
import resource
from .dataset import Dataset, TimeSeriesDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class SP500(TimeSeriesDataset):
    """The raw data is in https://archive.ics.uci.edu/dataset/554/cnnpred+cnn+based+stock+market+prediction+using+a+diverse+set+of+variables. 
        This dataset contains several daily features of S&P 500  from 2010 to 2017.
    """
    name:str= 'SP500'
    num_features: int = 70
    sample_rate:int # in hours
    length : int = 1984
    windows : int = 40
    freq:str = 'd'
    
    def download(self) -> None:
        # download_and_extract_archive(
        #     "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        #     self.dir,
        #     filename="traffic.txt.gz",
        #     md5="db745d0c9f074159581a076cbb3f23d6",
        # )
        pass
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'Processed_S&P.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=[0])
        self.df = self.df.drop(columns=['Name', 'mom', 'mom1', 'mom2', 'mom3', 'ROC_5', 'ROC_10', 'ROC_15','ROC_20', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200']).fillna(method='ffill').fillna(method='backfill')
        self.dates = pd.DataFrame({'date':self.df.Date})
        self.data = self.df.drop("Date", axis=1).values
        # self.data = np.loadtxt(self.file_name, delimiter=',')
        return self.data
        