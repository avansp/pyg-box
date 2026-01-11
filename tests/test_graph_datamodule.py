import pytest
from pathlib import Path
from src.modules.data.graph_datamodule import GraphDataModule
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path


def test_datamodule_mutag(tmp_path: Path):
    """Test instantiating GraphDataModule with TUDataset's MUTAG graph data."""
    data_dir = tmp_path / "data"
    print(f"{data_dir=}")
    name = "MUTAG"

    cfg = OmegaConf.create({
        "_target_": "src.modules.data.graph_datamodule.GraphDataModule",
        "batch_size": 64, 
        "dataset": {
            "_target_": "torch_geometric.datasets.TUDataset",
            "root": data_dir,
            "name": name
        }
    })

    dm : GraphDataModule = instantiate(cfg)

    assert Path(data_dir, name).exists()
    assert Path(data_dir, name, "raw").exists()
    assert Path(data_dir, name, "processed").exists()

    assert len(dm.train_dataloader()) > 0
    assert len(dm.val_dataloader()) > 0
    assert len(dm.test_dataloader()) > 0

