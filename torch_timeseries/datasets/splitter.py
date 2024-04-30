from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet


class Splitter:
    pass


class SequenceSplitter:
    def __init__(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.6,
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        num_worker: int = 3,
    ) -> None:
        """

        Split the dataset sequentially, and then randomly sample from each subset.

        :param dataset: the input dataset, must be of type datasets.Dataset
        :param train_ratio: the ratio of the training set
        :param test_ratio: the ratio of the testing set
        :param val_ratio: the ratio of the validation set
        """
        assert (
            train_ratio + test_ratio + val_ratio == 1.0
        ), "Split ratio must sum up to 1.0"
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_worker = num_worker

    def __getstate__(self) -> dict:
        # Only serialize the name and age members
        return {
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "val_ratio": self.val_ratio,
            "batch_size": self.batch_size,
        }

    def __setstate__(self, state: dict) -> None:
        # Restore the name and age members from the deserialized dictionary
        self.name = state["name"]
        self.age = state["age"]

    def __call__(self, dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """

        dataset_size = len(dataset)
        train_size = int(self.train_ratio * len(dataset))
        test_size = int(self.val_ratio * len(dataset))
        val_size = len(dataset) - test_size - train_size

        # fixed suquence dataset
        indices = range(0, len(dataset))
        train_dataset = Subset(dataset, indices[0:train_size])
        val_dataset = Subset(dataset, indices[train_size : (test_size + train_size)])
        test_dataset = Subset(dataset, indices[-val_size:])
        assert len(train_dataset) + len(test_dataset) + len(val_dataset) == dataset_size

        train_size = int(self.train_ratio * dataset_size)
        test_size = int(self.test_ratio * dataset_size)
        val_size = dataset_size - train_size - test_size
        assert (
            train_size + test_size + val_size == dataset_size
        ), "Data split sizes do not match the dataset size"

        # RandomSampler 与 Dataloader generator都需要设置，否则还是无法复现
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        return train_loader, test_loader, val_loader





class SequenceRandomSplitter:
    def __init__(
        self,
        seed=42,
        batch_size: int = 32,
        train_ratio: float = 0.6,
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        num_worker=4,
    ) -> None:
        """

        Split the dataset sequentially, and then randomly sample from each subset.

        :param dataset: the input dataset, must be of type datasets.Dataset
        :param train_ratio: the ratio of the training set
        :param test_ratio: the ratio of the testing set
        :param val_ratio: the ratio of the validation set
        """
        assert (
            train_ratio + test_ratio + val_ratio == 1.0
        ), "Split ratio must sum up to 1.0"
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle_test = shuffle_test
        self.shuffle_val = shuffle_val
        self.shuffle_train = shuffle_train
        self.num_worker = num_worker

    def __getstate__(self) -> dict:
        # Only serialize the name and age members
        return {
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "val_ratio": self.val_ratio,
            "batch_size": self.batch_size,
        }

    def __setstate__(self, state: dict) -> None:
        # Restore the name and age members from the deserialized dictionary
        self.name = state["name"]
        self.age = state["age"]

    def __call__(self, dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """

        seed_generator = torch.Generator()
        seed_generator.manual_seed(self.seed)

        dataset_size = len(dataset)
        train_size = int(self.train_ratio * len(dataset))
        test_size = int(self.test_ratio * len(dataset))
        val_size = len(dataset) - test_size - train_size

        # fixed suquence dataset
        indices = range(0, len(dataset))
        train_dataset = Subset(dataset, indices[0:train_size])
        val_dataset = Subset(dataset, indices[train_size : (test_size + train_size)])
        test_dataset = Subset(dataset, indices[-val_size:])
        assert len(train_dataset) + len(test_dataset) + len(val_dataset) == dataset_size

        train_size = int(self.train_ratio * dataset_size)
        test_size = int(self.test_ratio * dataset_size)
        val_size = dataset_size - train_size - test_size
        assert (
            train_size + test_size + val_size == dataset_size
        ), "Data split sizes do not match the dataset size"

        if self.shuffle_train:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=RandomSampler(train_dataset, generator=seed_generator),
                generator=seed_generator,
                num_workers=self.num_worker,
            )
        else:
            # RandomSampler 与 Dataloader generator都需要设置，否则还是无法复现
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False
            )

        if self.shuffle_val:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                sampler=RandomSampler(val_dataset, generator=seed_generator),
                generator=seed_generator,
                num_workers=self.num_worker,
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_worker,
            )

        if self.shuffle_test:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                sampler=RandomSampler(test_dataset, generator=seed_generator),
                generator=seed_generator,
                num_workers=self.num_worker,
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_worker,
            )
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        return train_loader, test_loader, val_loader
