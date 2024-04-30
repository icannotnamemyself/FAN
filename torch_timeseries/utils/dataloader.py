from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np


def dataloader_to_numpy(data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a PyTorch DataLoader object to a numpy array.

    Args:
        data_loader (DataLoader): The PyTorch DataLoader object to convert.

    Returns:
        A numpy array containing the data from the DataLoader.
    """
    data_numpy = []
    data_numpy_target = []
    for batch_x, batch_y in data_loader:
        data_numpy.append(batch_x.numpy())
        data_numpy_target.append(batch_y.numpy())
    data_numpy = np.concatenate(data_numpy, axis=0)
    data_numpy_target = np.concatenate(data_numpy_target, axis=0)
    return data_numpy, data_numpy_target

