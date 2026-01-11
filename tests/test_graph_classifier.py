from src.modules.model.graph_classifier import GraphClassifierModule
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
import pytest
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset


@pytest.mark.parametrize("num_classes,node_features", [(2, 1), (2, 2), (5, 7)])
def test_graph_classifier(num_classes: int, node_features: int):
    """Test creating GraphClassifierModule using hydra config."""

    with initialize(version_base="1.3", config_path="../src/configs/model"):
        cfg = compose(config_name="graph_classifier.yaml", overrides=[])
        cfg.num_classes = num_classes
        cfg.num_features = node_features

    model: GraphClassifierModule = instantiate(cfg)

    # test input size with FakeGraphDataset
    ds = FakeDataset(num_graphs=128, num_channels=node_features, num_classes=num_classes, task="graph")
    dataloader = DataLoader(ds, batch_size=64)
    
    for d in dataloader:
        # check forward
        y = model(d)
        assert y.size() == torch.Size((64, num_classes))

        # check model_step
        loss, preds, y = model.model_step(d)
        assert loss.dtype == torch.float32
        assert preds.size() == torch.Size([64])
        assert y.size() == torch.Size([64])
        assert (preds >= 0).bitwise_and(preds < num_classes).all()
