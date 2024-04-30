import os
import resource
from .dataset import Dataset, TimeSeriesDataset
from typing import Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np

class Traffic(TimeSeriesDataset):
    """The raw data is in http://pems.dot.ca.gov. 
    The data in this repo is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. 
    The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.
    """
    name:str= 'traffic'
    num_features: int = 862
    sample_rate:int # in hours
    length : int = 17544
    windows : int = 168
    freq:str = 'h'
    
    def download(self) -> None:
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
            self.dir,
            filename="traffic.txt.gz",
            md5="db745d0c9f074159581a076cbb3f23d6",
        )
        
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'traffic.txt')
        self.df = pd.read_csv(self.file_name, header=None)
        self.df['date'] = pd.date_range(start='01/01/2015 00:00', periods=self.length, freq='1H')  # '5T' for 5 minutes
        self.dates =  pd.DataFrame({'date': self.df['date'] })

        # self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop("date", axis=1).values
        # self.data = np.loadtxt(self.file_name, delimiter=',')
        return self.data
        