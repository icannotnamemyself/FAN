import torch
from torch import Tensor
from typing import Generic, TypeVar, Union

import pandas as pd
import numpy as np
import torch

StoreType = TypeVar(
    "StoreType", bound=Union[pd.DataFrame, np.ndarray, torch.Tensor]
)  # Type variable for input and output data


class Scaler(Generic[StoreType]):
    def fit(self, data: StoreType) -> None:
        """
        Fit the Scaler  to the input dataset.

        Args:
            data:
                The input dataset to fit the Scaler  to.
        Returns:
            None.
        """
        raise NotImplementedError()

    def transform(self, data: StoreType) -> StoreType:
        """
        Transform the input dataset using the Scaler .

        Args:
            data:
                The input dataset to transform using the Scaler .
        Returns:
            The transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()

    def inverse_transform(self, data: StoreType) -> StoreType:
        """
        Perform an inverse transform on the input dataset using the Scaler .

        Args:
            data:
                The input dataset to perform an inverse transform on.
        Returns:
            The inverse transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()


class MaxAbsScaler(Scaler[StoreType]):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self) -> None:
        self.scale = None

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.scale = np.max(np.abs(data), axis=0)
        elif isinstance(data, Tensor):
            self.scale = data.abs().max(axis=0).values
        else:
            raise ValueError(f"not supported type : {type(data)}")

    def transform(self, data) -> StoreType:
        # (b , n)  or (n)
        return data / self.scale

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.scale
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.scale, device=data.device)
        else:
            raise ValueError(f"not supported type : {type(data)}")

    # def __call__(self, tensor:Tensor):
    #     for ch in tensor:
    #         scale = 1.0 / (ch.max(dim=0)[0] - ch.min(dim=0)[0])
    #         ch.mul_(scale).sub_(ch.min(dim=0)[0])
    #     return tensor


class MinMaxScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    pass

class StandarScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, device="cpu") -> None:
        self.mean = None
        self.std = None
        self.device = device
        

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        elif isinstance(data, Tensor):
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
        else:
            raise ValueError(f"not supported type : {type(data)}")
        
        # print("Scaler:", self.mean, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.std + self.mean
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.std, device=data.device) + torch.tensor(
                self.mean, device=data.device
            )
        else:
            raise ValueError(f"not supported type : {type(data)}")



class NoScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, device="cpu") -> None:
        self.device = device

    def fit(self, data: StoreType):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data: StoreType) -> StoreType:
        return data
