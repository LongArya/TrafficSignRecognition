import os
from .data_split import DataSplit
from dataclasses import dataclass


@dataclass
class ModelLine:
    """Dataclass for keeping track of results produced by single training session"""

    root: str

    def __post_init__(self):
        os.makedirs(self.train_predictions_root, exist_ok=True)
        os.makedirs(self.val_predictions_root, exist_ok=True)
        os.makedirs(self.metrics_figures_root, exist_ok=True)
        os.makedirs(self.checkpoints_root, exist_ok=True)

    @property
    def train_predictions_root(self) -> str:
        return os.path.join(self.root, "train_predictions")

    @property
    def val_predictions_root(self) -> str:
        return os.path.join(self.root, "val_predictions")

    @property
    def test_predictions_root(self) -> str:
        return os.path.join(self.root, "test_predictions")

    @property
    def metrics_figures_root(self) -> str:
        return os.path.join(self.root, "metrics_figures")

    @property
    def checkpoints_root(self) -> str:
        return os.path.join(self.root, "checkpoints")

    def get_predictions_root_for_split(self, split: DataSplit) -> str:
        if split == DataSplit.TRAIN:
            return self.train_predictions_root
        elif split == DataSplit.VAL:
            return self.val_predictions_root
        elif split == DataSplit.TEST:
            return self.test_predictions_root
