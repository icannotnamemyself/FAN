import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import torch
from typing import Callable, List, Optional
import os
import resource
import numpy as np
from .dataset import Dataset, TimeSeriesDataset
from tqdm import tqdm


class SolarEnergy(TimeSeriesDataset):
    """The raw data is in http://www.nrel.gov/grid/solar-power-data.html :
    It contains the solar power production records in the year of 2006,
    which is sampled every 5 minutes from 137 PV plants in Alabama State.
    """

    name: str = "solar_AL"
    num_features: int = 137
    sample_rate: int  # in munites
    length: int = 52560
    freq:str = 'h'
    
    file_name = "solar_AL.txt"
    windows : int = 168

    def download(self) -> None:
        # download_and_extract_archive(
        #     "https://www.nrel.gov/grid/assets/downloads/al-pv-2006.zip",
        #     self.dir,
        #     filename="al-pv-2006.zip",
        #     md5="3fa6015aa550fc1f50d2f9bd6909403c",
        # )
        download_and_extract_archive(
            "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
            self.dir,
            filename="solar_AL.txt.gz",
            md5="41ef7fdc958c2ca3fac9cd06d6227073",
        )

    # def _check_procssed(self):
    #     self.file_path = os.path.join(self.dir, self.file_name)
    #     return check_integrity(self.file_path)

    # def _process(self) -> None:
    #     if not self._check_procssed():
    #         print(f"processing {self.name} files")
    #         all_files = os.listdir(self.dir)
    #         csv_files = [
    #             filename
    #             for filename in all_files
    #             if filename.endswith(".csv") and filename.endswith("5_Min.csv")
    #         ]
    #         df_list = []
    #         with tqdm(total=len(csv_files)) as progress_bar:
    #             for i, csv_file in enumerate(csv_files):
    #                 file_path = os.path.join(self.dir, csv_file)
    #                 df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    #                 column_name = f"Power{len(df_list)+1}(MW)"
    #                 df = df.rename(columns={"Power(MW)": column_name})
    #                 df_list.append(df[column_name])

    #                 progress_bar.update(1)
    #             merged_df = pd.concat(df_list, axis=1)
    #             merged_df.to_csv(os.path.join(self.dir, self.file_name))
    #     else:
    #         print("Using processed data ...")

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, self.file_name)
        self.df = pd.read_csv(self.file_path, sep=',', header=None) # pd.read_csv(self.file_path, parse_dates=["LocalTime"])
        # self.df = df.rename(columns={"LocalTime": "date"})
        self.df['date'] = pd.date_range(start='01/01/2006 00:00', periods=self.length, freq='10T')  # '5T' 是5分钟的意思
        self.dates =  pd.DataFrame({'date': self.df['date'] })
        self.data = self.df.drop("date", axis=1).values
        return self.data
