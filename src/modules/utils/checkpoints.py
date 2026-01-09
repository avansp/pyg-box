import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from typing import Optional, Union
from typing_extensions import override
import logging
from torch import Tensor
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class BestModelCheckpoint(ModelCheckpoint):
    """
    This class is especially useful for cases where you want to automatically save and track
    the model's best-performing states during training based on a specific metric. It provides
    support for operations such as saving checkpoints at the end of training epochs, validation
    stages, and finalizing the best model checkpoints at the end of the training run. The saved
    checkpoints can be named and managed conveniently for model reproducibility and deployment.

    Attributes:
        CHECKPOINT_BEST_MODEL: A constant string used to denote the name for the best model
            checkpoints.
    """
    CHECKPOINT_BEST_MODEL = "best_model"

    def __init__(
            self,
            dirpath: Union[str, Path],
            filename: Optional[str] = "ep{epoch:04d}_vl{val/loss:.4f}_tl{train/loss:.4f}",
            monitor: Optional[str] = "val/loss",
            mode: str = "min",
            save_top_k: Optional[int] = 1,
            verbose: bool = False
    ):
        super().__init__(
            dirpath = dirpath,
            filename = filename,
            monitor = monitor,
            every_n_train_steps = 0,
            save_last = False,
            auto_insert_metric_name = False,
            mode = mode,
            save_top_k = save_top_k,
            verbose = verbose
        )

        self.CHECKPOINT_EQUALS_CHAR = ""
        self.CHECKPOINT_JOIN_CHAR = "_"

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Handles operations to be executed at the end of the training. This method is an
        override of the default `on_train_end` method to include custom checkpointing
        logic. It saves the last checkpoint and creates a symbolic link for easier
        access to the latest checkpoint in the training process.

        Parameters:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model instance whose
                training is being managed.

        Returns:
            None
        """
        monitor_candidates = self._monitor_candidates(trainer)

        # save the last check point
        filepath = self.format_checkpoint_name(monitor_candidates, self.filename)
        logging.info(f"Saving last checkpoint to {Path(filepath).name}")
        self._save_checkpoint(trainer, filepath)

        # and create a link
        link_last_filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
        self._link_checkpoint(trainer, filepath, link_last_filepath)

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            self.save_best_model(trainer, monitor_candidates)

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            self.save_best_model(trainer, monitor_candidates)

    def save_best_model(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        """
        Saves the best model based on the monitored metric. This method evaluates the monitored metric in the given
        `monitor_candidates` dictionary and saves the model if it meets the conditions defined for the configuration
        of the monitoring. It updates the best score and performs the save operation for the model if necessary.

        Parameters:
        monitor_candidates : dict[str, Tensor]
            A dictionary containing the metrics logged during training. Each key is the name of a metric,
            and the value is its corresponding value as a Tensor.

        trainer : pl.Trainer
            The Trainer object managing the training, used for accessing state and saving the model.

        Raises:
        MisconfigurationException
            If the specified monitored metric is not found in the `monitor_candidates` dictionary.
        """
        if self.monitor is None:
            return

        # validate metric
        if self.monitor not in monitor_candidates:
            m = (
                f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                f" metrics: {list(monitor_candidates)}."
                f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
            )
            raise MisconfigurationException(m)

        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)

            # save the metric too
            train_result = trainer.callback_metrics
            with open(Path(self.dirpath, "best_metric.csv"), "w") as fout:
                fout.write("dataset, metric, value\n")
                for k, v in train_result.items():
                    dm = k.split('/')
                    fout.write(f"{dm[0]}, {dm[1]}, {v}\n")

    @override
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not Path(self.best_model_path).is_file():
            return
        
        assert Path(self.best_model_path).is_file(), f"Best model path {self.best_model_path} does not exist!"

        logging.info(f"Best checkpoint path: {Path(self.best_model_path).name}")

        # create a link for the best checkpoint path
        best_path = self.format_checkpoint_name(self._monitor_candidates(trainer), self.CHECKPOINT_BEST_MODEL)
        self._link_checkpoint(trainer, self.best_model_path, best_path)

        # save for the last metric too
        train_result = trainer.callback_metrics
        with open(Path(self.dirpath, "last_metric.csv"), "w") as fout:
            fout.write("dataset, metric, value\n")
            for k, v in train_result.items():
                dm = k.split('/')
                fout.write(f"{dm[0]}, {dm[1]}, {v}\n")


