import json
import os
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torch


class DummyInferenceCallback(Callback):
    """Callback that runs model on dummy input in order to assert reproducibility later"""

    def __init__(self, dummy_input: torch.Tensor, save_root: str):
        super().__init__()
        self.dummy_input = dummy_input
        self.save_root = save_root

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logits = pl_module.model(self.dummy_input.to(pl_module.device))
        json_serializible_logits = logits.cpu().tolist()
        output_path: str = os.path.join(
            self.save_root, f"{pl_module.current_epoch:04d}.json"
        )
        with open(output_path, "w") as f:
            json.dump({"dummy_input_logits": json_serializible_logits}, f, indent=2)
        # save logits to neptune
        return super().on_validation_epoch_end(trainer, pl_module)
