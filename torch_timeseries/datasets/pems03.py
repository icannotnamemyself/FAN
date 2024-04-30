import os
import pickle
from.dataset import Dataset, Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset
from typing import Any, Callable, List, Optional
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data
from torch_timeseries.utils.load_pickle import load_pickle



class PEMS03(TimeSeriesStaticGraphDataset):
    name:str= 'PEMS03'
    num_features: int = 358
    length : int  = 26209
    freq : Freq = 'h'
    windows : int = 12
    
    def download(self):
        """download from https://github.com/Davidham3/STSGCN
        """
        pass
        
    def _load(self) -> np.ndarray:
        self._load_static_graph()
        self.file_path = os.path.join(self.dir, 'PEMS03_data.csv')
        self.df = pd.read_csv(self.file_path, header=None)
        self.df['date'] = pd.date_range(start='9/1/2018 00:00', periods=self.length, freq='5T')  # '5T' for 5 minutes
        self.dates =  pd.DataFrame({'date': self.df['date'] })
        self.data = self.df.drop("date", axis=1).values
        return self.data
    
    def _load_static_graph(self):
        sensor_ids = np.loadtxt(os.path.join(self.dir, 'PEMS03.txt')) 
        distances = pd.read_csv(os.path.join(self.dir, 'PEMS03.csv')).to_numpy()
        num_sensors = len(sensor_ids)
        adj_matrix = np.zeros((num_sensors, num_sensors))

        id_to_index = {sid: idx for idx, sid in enumerate(sensor_ids)}

        for from_id, to_id, distance in distances:
            if from_id in id_to_index and to_id in id_to_index:
                from_index = id_to_index[from_id]
                to_index = id_to_index[to_id]
                adj_matrix[from_index, to_index] = distance
                adj_matrix[to_index, from_index] = distance  

        self.adj = adj_matrix
        
    
    