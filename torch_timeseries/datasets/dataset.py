from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, Sequence, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod

from torch_timeseries.data.scaler import MaxAbsScaler, Scaler, StoreType


from enum import Enum, unique


@unique
class Freq(str, Enum):
    seconds = "s"
    minutes = "t"
    hours = "h"
    days = "d"
    months = "m"
    years = "y"


class Dataset(torch.utils.data.Dataset):
    name: str
    num_features: int
    length: int
    freq: Freq

    def __init__(self, root: str):
        """_summary_

        Args:
            root (str): data save location
            transform (Optional[Callable], optional): _description_. Defaults to None.
            target_transform (Optional[Callable], optional): _description_. Defaults to None.
            single_step (bool, optional): True for single_step data, False for multi_steps data. Defaults to True.
        """
        super().__init__()

        self.data: np.ndarray
        self.df: pd.DataFrame

    def download(self):
        r"""Downloads the dataset to the :obj:`self.dir` folder."""
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


# StoreTypes = Union[np.ndarray, Tensor]
StoreTypes = np.ndarray


class TimeSeriesDataset(Dataset):
    def __init__(self, root: str='./data'):
        """_summary_

        Args:
            root (str): data save location

        """
        super().__init__(root)
        self.root = root
        self.dir = os.path.join(root, self.name)
        os.makedirs(self.dir, exist_ok=True)
        
        self.download()
        self._process()
        self._load()
        
        self.dates: Optional[pd.DataFrame ]
        

    @abstractmethod
    def download(self):
        r"""Downloads the dataset to the :obj:`self.dir` folder."""
        raise NotImplementedError

    def _process(self) :
        pass
    
    
    @abstractmethod
    def _load(self) -> StoreTypes:
        """Loads the dataset to the :attr:`self.data` .

        Raises:
            NotImplementedError: _description_

        Returns:
            T: should return a numpy.array or torch.tensor or pandas.Dataframe
        """

        raise NotImplementedError


class TimeSeriesStaticGraphDataset(TimeSeriesDataset):
    adj : np.ndarray 
    def _load_static_graph(self):
        raise NotImplementedError()


class TimeseriesSubset(torch.utils.data.Subset):
    def __init__(self, dataset: TimeSeriesDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.data = self.dataset.data[indices]
        self.df = self.dataset.df.iloc[indices]
        self.dates = self.dataset.dates.iloc[indices]
        self.num_features = dataset.num_features
        self.name = dataset.name
        self.length = len(self.indices)
        self.freq = dataset.freq
