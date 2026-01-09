"""This file prepares files and configurations for unit tests."""

import pytest
from omegaconf import DictConfig, open_dict
from hydra import compose, initialize
import rootutils
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(scope="function")
def cfg_train_mnist(tmp_path: Path) -> DictConfig:
    """A pytest fixture to setup configuration for task=mnist."""
    root_dir = rootutils.find_root(indicator=".project-root")

    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, 
                      overrides=["task=mnist", "trainer=sanity_check"])
        cfg.paths.root_dir = str(root_dir)
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    return cfg

