import os
import pickle
from.dataset import Dataset, Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset
from typing import Any, Callable, List, Optional
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data
from torch_timeseries.utils.load_pickle import load_pickle



class PEMS08(TimeSeriesStaticGraphDataset):
    name:str= 'PEMS08'
    num_features: int = 170
    length : int  = 17856
    freq : Freq = 'h'
    windows : int = 12
    
    def download(self):
        """download from https://github.com/Davidham3/STSGCN
        """
        pass
        
    def _load(self) -> np.ndarray:
        self._load_static_graph()
        with np.load(os.path.join(self.dir, 'PEMS08.npz')) as data:
            np_data =  data['data'][:,:,0]

        self.df = pd.DataFrame(data=np_data, columns=list(range(np_data.shape[1])))
        self.df['date'] = pd.date_range(start='7/1/2016 00:00', periods=self.length, freq='5T')  # '5T' for 5 minutes
        self.dates =  pd.DataFrame({'date': self.df['date'] })
        self.data = self.df.drop("date", axis=1).values
        return self.data
    
    def _load_static_graph(self):
        with np.load(os.path.join(self.dir, 'PEMS08.npz')) as data:
            num_sensors =  data['data'].shape[1]

        distances = pd.read_csv(os.path.join(self.dir, 'PEMS08.csv')).to_numpy()
        adj_matrix = np.zeros((num_sensors, num_sensors))


        # 步骤 3: 填充邻接矩阵
        for from_id, to_id, distance in distances:
            adj_matrix[int(from_id), int(to_id)] = distance
            adj_matrix[int(to_id), int(from_id)] = distance  # 如果是无向图，反方向也是相同的距离

        self.adj = adj_matrix
        
    
    