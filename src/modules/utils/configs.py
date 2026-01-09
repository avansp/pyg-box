from omegaconf import OmegaConf, DictConfig
from pathlib import Path


def load_config(config_root: Path | str, filename: str = "config.yaml") -> DictConfig:
    """
    Loads a configuration file and returns it as a dictionary-like structure.

    This function reads a YAML configuration file from the specified directory,
    parses it, and returns its contents as an `OmegaConf.DictConfig` object.
    The presence of the configuration file is verified before attempting to
    load and parse it.

    Args:
        config_root (Path): The path to the directory containing the configuration file.
        filename (str): The name of the configuration file. Defaults to "config.yaml".

    Returns:
        DictConfig: The parsed configuration represented as a dictionary-like object.
    """
    config_file = Path(config_root) / filename
    assert config_file.is_file(), f"Config file not found at {config_file}"

    with open(config_file, "r") as f:
        cfg = OmegaConf.load(f)

    return cfg
