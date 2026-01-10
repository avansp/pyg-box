from torch_geometric.data import Dataset
from src.modules.data.data_splitter import BaseSplitter, RandomSplitter
from torch_geometric.data.lightning import LightningDataset


class GraphDataModule(LightningDataset):
    def __init__(
        self,
        dataset: Dataset,
        splitter: BaseSplitter = RandomSplitter(split=[0.6, 0.2, 0.2])
    ) -> None:
        # prepare the data_train, data_val and data_test
        dataset.shuffle()

        # run the splitting based on the given splitter method
        data_train, data_val, data_test = splitter.run(dataset)

        super().__init__(
            train_dataset=data_train,
            val_dataset=data_val,
            test_dataset=data_test
        )


    