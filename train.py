import os
from dataset import RussianTrafficSignBaseDataset, SquareCropReader, TransformApplier
from pprint import pprint
from classification import TrafficSignsClassifier, ModelLine, TrafficSignTrainerConfig
from classification.rtsd_classifier import init_augmentations_from_config
from callbacks import DummyInferenceCallback
from torch.utils.data import Dataset, Subset, default_collate, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from typing import List, Dict, Any
import clearml
from clearml import Task
import hydra
from loggers.clearml_logger import ClearMLLogger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
TRAIN_RESULTS_ROOT = os.path.join(WORKSPACE_DIR, "training_results")
CFG_ROOT = os.path.join(WORKSPACE_DIR, "code", "static_gesture_classification", "conf")
CFG_NAME = "base_traffic_sign_config"


def load_dummy_dataset():
    val_coco_file: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\val_anno.json"
    images_root: str = "E:\\ITMO\\TrafficSignsCV\\archive (1)\\rtsd-frames"
    dataset = RussianTrafficSignBaseDataset(
        coco_ann_file=val_coco_file, images_root=images_root
    )
    dataset = SquareCropReader(dataset)
    dataset = Subset(dataset, indices=list(range(10)))


def load_train_dataset():
    load_dummy_dataset()


def load_val_dataset():
    load_dummy_dataset()


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_batch: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = default_collate([sample[key] for sample in batch])
    return collated_batch


def init_clearml_task() -> Task:
    pass


@hydra.main(
    config_path=CFG_ROOT,
    config_name=CFG_NAME,
    version_base=None,
)
def train_static_gesture(cfg: TrafficSignTrainerConfig):
    run_id: str = ""
    model_line_path: str = os.path.join(TRAIN_RESULTS_ROOT, run_id)
    model_line: ModelLine = ModelLine(model_line_path)
    lightning_classifier = TrafficSignTrainerConfig(cfg, results_location=model_line)

    # log_dict_like_structure_to_neptune(
    #     dict_like_structure=cfg,
    #     neptune_root="conf",
    #     neptune_run=neptune_logger.experiment,
    #     log_as_sequence=False,
    # )
    task = Task()
    train_dataset = load_train_dataset()
    val_dataset = load_val_dataset()
    augmentations = init_augmentations_from_config(augs_cfg=cfg.augs)

    train_dataset = TransformApplier(
        dataset=train_dataset, transformation=augmentations[DataSplit.TRAIN]
    )
    val_dataset = TransformApplier(
        dataset=val_dataset, transformation=augmentations[DataSplit.VAL]
    )
    # log_info_about_datasets_to_neptune(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     neptune_run=neptune_logger.experiment,
    #     logged_samples_amount=10,
    #     augs_config=cfg.augs,
    # )
    dummy_output_directory = os.path.join(model_line.root, "dummy_output")
    os.makedirs(dummy_output_directory, exist_ok=True)

    dummy_inference_callback = DummyInferenceCallback(
        dummy_input=torch.zeros(1, 3, *cfg.augs.input_resolution),
        save_root=dummy_output_directory,
    )
    lr_monitor_callback = LearningRateMonitor(
        logging_interval="step", log_momentum=True
    )
    model_ckpt_callback = ModelCheckpoint(
        monitor="val_weighted_F1",
        dirpath=model_line.checkpoints_root,
        mode="max",
        auto_insert_metric_name=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        filename="checkpoint_{epoch:02d}-{val_weighted_F1:.2f}",
        save_top_k=3,
    )
    logger = ClearMLLogger(clearml_logger=task._logger)
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=1,
        max_epochs=100,
        callbacks=[dummy_inference_callback, model_ckpt_callback, lr_monitor_callback],
        num_sanity_val_steps=0,
        gpus=[0],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=custom_collate,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=custom_collate,
        shuffle=False,
    )
    trainer.fit(
        model=lightning_classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path="E:\\dev\\MyFirstDataProject\\training_results\\STAT-90\\checkpoints\\checkpoint_epoch=48-val_weighted_F1=0.87.ckpt",
    )


if __name__ == "__main__":
    train_static_gesture()
