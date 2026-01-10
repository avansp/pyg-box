from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from torch.utils.data import random_split, Subset
from torch_geometric.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


class BaseSplitter(ABC):
    def __init__(self, split: list):
        self.split = split
        super().__init__()

        assert len(self.split) == 3, f"Split must contain of 3 elements for [train, validate, test] subset."

    @abstractmethod
    def run(self, data: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        pass


class FixedSplitter(BaseSplitter):
    """Split has been fixed by the split indices.

    split is [[train_idx],[val_idx],[test_idx]]

    Example:
        >>> sp = FixedSplitter(split=[[0,1,2],[3,4],[5]])
    """
    def __init__(self, split: List):
        super().__init__(split=split)

    def run(self, data: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        train_dset = data.copy().index_select(self.split[0])
        val_dset = data.copy().index_select(self.split[1])
        test_dset = data.copy().index_select(self.split[2])

        return train_dset, val_dset, test_dset


class RandomSplitter(BaseSplitter):
    """Split dataset randomly.

    Example:
        >>> sp = RandomSplitter(split=[0.2, 0.3, 0.5])
    """
    def run(self, data: Dataset) -> tuple[Subset[Any], Subset[Any], Subset[Any]]:
        train, val, test = random_split(data, self.split)

        return train, val, test


class StratifiedSplitter(BaseSplitter):
    """Split dataset using StratifiedShuffleSplit

    Example:
        >>> sp = StratifiedSplitter(split=[0.6, 0.1, 0.3])

    See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    """
    @staticmethod
    def do_split(data: Dataset, split_size: float):
        # get class list
        # y = [ds.y for ds in data]
        y = [ds.y if not isinstance(ds.y, list) else tuple(ds.y) for ds in data]

        # split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split_size)
        train_idx, test_idx = next(sss.split(X=[0] * len(y), y=y))

        return train_idx, test_idx

    def run(self, data: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        # we're going to use StratifiedShuffleSplit twice
        # for the test, we use the split directly
        train_idx, test_idx = self.do_split(data, self.split[2])

        # define the test dataset
        test_dataset = data.copy().index_select(test_idx)

        # next split for train & val
        sub_data = data.index_select(train_idx).copy()
        train_idx, val_idx = self.do_split(sub_data, self.split[1])

        # define the train dataset
        train_dataset = sub_data.copy().index_select(train_idx)

        # define the validation dataset
        val_dataset = sub_data.copy().index_select(val_idx)

        return train_dataset, val_dataset, test_dataset

