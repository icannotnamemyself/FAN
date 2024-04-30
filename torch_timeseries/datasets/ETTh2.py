import os
import resource
from .dataset import Dataset, Freq, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import (
    download_url,
    download_and_extract_archive,
    check_integrity,
)
import pandas as pd
import numpy as np
import torch.utils.data


class ETTh2(TimeSeriesDataset):
    name: str = "ETTh2"
    freq: Freq = "h"
    num_features: int = 7
    length: int = 17420
    windows : int = 384

    def download(self):
        download_url(
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
            self.dir,
            filename="ETTh2.csv",
            md5="51a229a3fc13579dd939364fefe9c7ab",
        )

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, "ETTh2.csv")
        self.df = pd.read_csv(self.file_path, parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data

    def _process(self) -> np.ndarray:
        return super()._process()
