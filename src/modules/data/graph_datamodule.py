from torch_geometric.data import Dataset
from src.modules.data.data_splitter import BaseSplitter, RandomSplitter
from torch_geometric.data.lightning import LightningDataset
from typing import Any


class GraphDataModule(LightningDataset):
    """
    A PyTorch Lightning DataModule for handling graph datasets with customizable splitting strategies.

    This class extends LightningDataset to manage train, validation, and test splits for graph data.
    It shuffles the input dataset and applies a specified splitting strategy to partition the data
    into training, validation, and test sets.

    Args:
        dataset (Dataset): The graph dataset to be split and managed.
        splitter (BaseSplitter, optional): The strategy used to split the dataset into train/val/test sets.
            Defaults to RandomSplitter with split ratios [0.6, 0.2, 0.2] for train/val/test respectively.
        **kwargs (Any): Additional keyword arguments to pass to the parent LightningDataset class.

    Attributes:
        train_dataset: The training subset of the dataset.
        val_dataset: The validation subset of the dataset.
        test_dataset: The test subset of the dataset.

    Example:
        >>> dataset = MyGraphDataset()
        >>> datamodule = GraphDataModule(dataset, splitter=RandomSplitter(split=[0.7, 0.15, 0.15]))
    """
    def __init__(
        self,
        dataset: Dataset,
        splitter: BaseSplitter = RandomSplitter(split=[0.6, 0.2, 0.2]),
        **kwargs: Any
    ) -> None:
        # prepare the data_train, data_val and data_test
        dataset.shuffle()

        # run the splitting based on the given splitter method
        data_train, data_val, data_test = splitter.run(dataset)

        super().__init__(
            train_dataset=data_train,
            val_dataset=data_val,
            test_dataset=data_test,
            **kwargs
        )


    