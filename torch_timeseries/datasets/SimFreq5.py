

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

from torch_timeseries.datasets.dataset import Freq, TimeSeriesDataset, TimeSeriesStaticGraphDataset

class SimFreq5(TimeSeriesDataset):
    name: str = 'SimFreq5'
    num_features:int = 5
    sample_rate:int = 1
    length : int= 10000
    freq: str = 't'
    
    def download(self): 
        pass
    

    def _load(self):
        # n = 400
        # Generating date series
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq='t')
        
        # Creating a data matrix
        data = np.zeros((len(dates), self.num_features))
        
        # freqs_mag = ([(0,1,2,4),(1,3,5,6),(3,4,6,8),(1,2,4,5)])
        # freqs = (12,24,48,72)
        
        freqs_mag = ([(0,1,2,4),(1,3,5,6),(3,4,6,8),(1,2,4,5),(1,3,5,6)])
        freqs = (12,24,48,72,96)

        
        seqs = []
        for i in range(self.num_features):
            seqs_ = []
            seqs_.append(np.linspace(freqs_mag[i][0],freqs_mag[i][1], num=int(self.length*0.7)))
            seqs_.append(np.linspace(freqs_mag[i][1], freqs_mag[i][2], num=int(self.length*0.2)))
            seqs_.append(np.linspace(freqs_mag[i][2], freqs_mag[i][3] , num=int(self.length) - int(self.length*0.7) - int(self.length*0.2) ) )
            seqs.append( np.concatenate(seqs_))
        # seq1 = np.linspace(1, 3, num=int(self.length*0.7))
        # seq2 = np.linspace(3, 2.5, num=int(self.length*0.2))
        # seq3 = np.linspace(2.5, 4 , num=int(self.length) - len(seq2) - len(seq1) )
        # combined_seq = np.concatenate((seq1, seq2, seq3))

        for i in range(0, self.num_features):
            t = np.arange(0, len(dates))
            x = 0
            
            freq_signals  = 0 
            for j in range(0, i+1):
                w = 2*np.pi / freqs[j]
                freq_signals += np.sin(w * t)
            x = seqs[j]*freq_signals
            # x += +  (t // n)
            data[:, i] = x

        # Creating DataFrame with specified column names
        self.df = pd.DataFrame(data, columns=[ f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates =  pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values        
        return self.data

