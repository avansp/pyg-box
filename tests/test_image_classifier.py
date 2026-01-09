from src.modules.model.nets.dense_net import SimpleDenseNet
from src.modules.model.image_classifier import ImageClassifierLitModule
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
import pytest


@pytest.mark.parametrize("channel", [1, 3])
def test_image_classifier(channel: int):
    """Test creating ImageClassifierLitModule using hydra config."""
    width = 100
    height = 120
    output_size = 5

    with initialize(version_base="1.3", config_path="../src/configs/model"):
        cfg = compose(config_name="image_classifier.yaml", overrides=[])
        cfg.width = width
        cfg.height = height
        cfg.channel = channel
        cfg.num_classes = output_size

    model: ImageClassifierLitModule = instantiate(cfg)

    # test input size with channel
    y = model(torch.rand(28, channel, width, height))
    assert y.size() == torch.Size([28, output_size])

    # test model step
    loss, preds, y = model.model_step([
        torch.rand(50, channel, width, height), 
        torch.randint(low=0, high=output_size-1, size=(50, output_size), dtype=torch.float32)])
    
    assert loss.dtype == torch.float32
    assert loss.size() == torch.Size([])
    assert preds.size() == torch.Size([50])
    assert y.size() == torch.Size([50, output_size])
    assert (preds >= 0).bitwise_and(preds < output_size).all()

