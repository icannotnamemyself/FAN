

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

class Dummy(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 8
    sample_rate:int = 1
    length : int= 1000
    def download(self): 
        pass
    
    def _load(self):
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        # 创建一个数据矩阵
        data = np.random.rand(len(dates), 2)

        # 将时间列和数据矩阵拼接成一个numpy数组
        result = np.concatenate([dates[:, np.newaxis], data], axis=1)

        # 创建DataFrame，指定列名
        self.df = pd.DataFrame(result, columns=['date', 'data1', 'data2'])
        self.data = self.df.drop('date').values        
        return self.data


class DummyDatasetGraph(TimeSeriesStaticGraphDataset):
    name: str = 'dummy_graph'
    num_features:int = 5
    freq : Freq = Freq.minutes
    length : int = 1440
    
    def _load_static_graph(self):
        self.adj = np.ones((self.num_features, self.num_features))
    def download(self): 
        pass
    
    def _load(self):
        self._load_static_graph()
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01',periods=self.length, freq='t')

        # 创建一个数据矩阵
        data = np.random.rand(len(dates), self.num_features)
        # 将时间列和数据矩阵拼接成一个numpy数组
        result = {'date': dates}
        # iterate to get above df
        # 循环遍历每一列数据并添加到字典
        for i in range(data.shape[1]):  # 假设 data 是一个 NumPy 数组
            key = f'data{i+1}'  # 创建键，例如 'data1', 'data2', ...
            result[key] = data[:, i]  # 添加数据到字典
        self.df = pd.DataFrame(result)

        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop('date', axis=1).values        
        return self.data


class DummyWithTime(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 2
    freq : Freq = Freq.minutes
    length : int = 1440
    def download(self): 
        pass
    
    def _load(self):
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        # 创建一个数据矩阵
        data = np.random.rand(len(dates), 2)
        # 将时间列和数据矩阵拼接成一个numpy数组
        self.df = pd.DataFrame({'date': dates, 'data1': data[:, 0],'data2': data[:, 1]})
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop('date', axis=1).values        
        return self.data
    
    
    
    
class DummyWithTime(TimeSeriesDataset):
    name: str = 'dummy'
    num_features:int = 2
    freq : Freq = Freq.minutes
    length : int = 1440
    def download(self): 
        pass
    
    def _load(self):
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01', end='2022-01-02', freq='t')

        # 创建一个数据矩阵
        data = np.random.rand(len(dates), 2)
        # 将时间列和数据矩阵拼接成一个numpy数组
        self.df = pd.DataFrame({'date': dates, 'data1': data[:, 0],'data2': data[:, 1]})
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop('date', axis=1).values        
        return self.data






class DummyContinuous(TimeSeriesDataset):
    name: str = 'dummy'
    num_features: int = 10
    freq: Freq = Freq.minutes
    length: int = 10000

    def download(self):
        pass
    
    def _load(self):
        # 生成日期序列
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq='t')
        # dates = pd.date_range(start='2022-01-01', end='2022-01-03', freq='t')
        
        # 初始化数据矩阵
        data = np.zeros((len(dates), self.num_features))
        
        # 为序列的最初三个时间点设置初始值，这里使用0作为示例
        data[:3, :] = np.random.rand(3, self.num_features)  # 可以选择其他初始值
        
        # 根据公式计算序列的其余部分
        for i in range(3, len(dates)):
            for j in range(self.num_features):  # 对每一列（特征）应用公式
                data[i, j] = (data[i-1, j]+ data[i-2, j])/data[i-3, j] + np.sqrt(i)
        
        # 将时间列和数据矩阵拼接成DataFrame
        self.df = pd.DataFrame(data, columns=[ f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates = pd.DataFrame({'date': self.df.date})
        self.data = self.df.drop('date', axis=1).values
        
        return self.data
