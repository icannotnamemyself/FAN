import os
import pickle
import resource
from.dataset import Dataset, Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data
from torch_timeseries.utils.load_pickle import load_pickle



class METR_LA(TimeSeriesStaticGraphDataset):
    name:str= 'METR-LA'
    num_features: int = 207
    length : int  = 34272
    freq : Freq = 'h'
    windows : int = 288
    
    def download(self):
        # download_url(
        #  "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/METR-LA/metr-la.h5",
        #  self.dir,
        #  filename='metr-la.h5',
        #  md5="92622ec9970a66de022b1aea1e567d16"
        # )
        pass
        
    def _load(self) -> np.ndarray:
        self._load_static_graph()
        self.file_path = os.path.join(self.dir, 'metr-la.h5')
        self.df = pd.read_hdf(self.file_path)
        self.df.index = self.df.index.map(lambda x: pd.Timestamp(str(x)))
        self.dates = pd.DataFrame({'date': pd.Series(self.df.index) })
        self.data = self.df.iloc[:, :].to_numpy()
        return self.data
    
    
    def _load_static_graph(self):
        self.adj = load_pickle(os.path.join(self.dir, 'adj_mx.pkl'))[2]
        
