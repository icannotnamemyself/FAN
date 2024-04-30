import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_url
import torch
from typing import Any, Callable, List, Optional
import os
import resource
from.dataset import Dataset, TimeSeriesDataset


class ExchangeRate(TimeSeriesDataset):
    name: str = 'exchange_rate'
    num_features: int = 8
    freq: str = 'd' # daily data
    length : int = 7588
    windows : int = 48

    
    
    def download(self) -> None:
        download_url(
            "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/exchange_rate/exchange_rate.csv",
            self.dir,
            filename="exchange_rate.csv",
            md5="2fc11972378a4c8817c1adfdde522bf9",
        )

    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'exchange_rate.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=['date'])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop("date", axis=1).values
        return self.data
