import pytest
from pathlib import Path
from src.modules.data.graph_datamodule import GraphDataModule
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path


def test_graph_datamodule():
    """Testing instantiating GraphDataModule and check loaders."""
    cfg = OmegaConf.create({
        "_target_": "src.modules.data.graph_datamodule.GraphDataModule",
        "batch_size": 15,
        "dataset": {
            "_target_": "torch_geometric.datasets.FakeDataset",
            "num_graphs": 300,
            "num_classes": 10
        },
        "splitter": {
            "_target_": "src.modules.data.data_splitter.RandomSplitter",
            "split": [0.6, 0.3, 0.1]
        }
    })

    data_module : GraphDataModule = instantiate(cfg)

    assert len(data_module.train_dataset) == 180
    assert len(data_module.val_dataset) == 90
    assert len(data_module.test_dataset) == 30

    for d in data_module.train_dataloader():
        assert len(d) == 15

    for d in data_module.val_dataloader():
        assert len(d) == 15

    for d in data_module.test_dataloader():
        assert len(d) == 15

