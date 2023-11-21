import os
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from dataset import (
    RussianTrafficSignBaseDataset,
    SquareCropReader,
    TransformApplier,
    LabelEnumApplier,
)
from pprint import pprint
from classification import (
    TrafficSignsClassifier,
    ModelLine,
    TrafficSignTrainerConfig,
    DataSplit,
)
from classification.rtsd_classifier import init_augmentations_from_config
from callbacks import DummyInferenceCallback
from torch.utils.data import Dataset, Subset, default_collate, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from typing import List, Dict, Any
import clearml
from clearml import Task
import hydra
from hydra.core.config_store import ConfigStore
from loggers.clearml_logger import ClearMLLogger
import torch
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
CFG_ROOT = os.path.join(SCRIPT_DIR, "conf")
CFG_NAME = "traffic_sign_classification_config"
TRAIN_RESULTS_ROOT = os.path.join(WORKSPACE_DIR, "training_results")
RTSD_DS_ROOT = os.path.join(WORKSPACE_DIR, "archive")
print(f"RTSD_DS_ROOT = {RTSD_DS_ROOT}")
RTSD_IMG_ROOT = os.path.join(RTSD_DS_ROOT, "rtsd-frames")
VAL_ANN_FILE = os.path.join(RTSD_DS_ROOT, "val_anno.json")
TRAIN_ANN_FILE = os.path.join(RTSD_DS_ROOT, "train_anno.json")
RTSD_LABELS_MAP_FILE = os.path.join(RTSD_DS_ROOT, "label_map.json")
BG_NUM_IN_DATASET = 100
MAX_AMOUNT_OF_SAMPLES_OF_ONE_CLASS = 1000

os.environ["HYDRA_FULL_ERROR"] = "1"

cs = ConfigStore.instance()
cs.store(name=CFG_NAME, node=TrafficSignTrainerConfig)


def load_dummy_dataset(label_enum: Dict[int, str]) -> Dataset:
    dataset = RussianTrafficSignBaseDataset(
        coco_ann_file=VAL_ANN_FILE,
        images_root=RTSD_IMG_ROOT,
        label_map_file=RTSD_LABELS_MAP_FILE,
    )
    dataset = LabelEnumApplier(dataset, label_enum_id2name=label_enum)
    dataset = SquareCropReader(dataset)
    dataset = Subset(dataset, indices=list(range(10)))
    return dataset


def limit_bg_num_in_dataset(dataset: Dataset, bg_num: int) -> Dataset:
    used_indexes: List[int] = []
    bg_indexes: List[int] = []
    for index, sample in enumerate(tqdm(dataset)):
        if sample["class_name"] != "background":
            used_indexes.append(index)
        else:
            bg_indexes.append(index)
    used_indexes.extend(bg_indexes[:bg_num])
    return Subset(dataset, used_indexes)


def limit_class_num_in_dataset(dataset: Dataset, max_class_presence: int) -> Dataset:
    classes_presence_indexes = defaultdict(list)
    for index, sample in enumerate(tqdm(dataset)):
        classes_presence_indexes[sample["class_name"]].append(index)
    used_indexes = []
    for class_name, class_indexes in classes_presence_indexes.items():
        used_indexes.extend(class_indexes[:max_class_presence])
    return Subset(dataset, used_indexes)


def load_train_dataset(label_enum: Dict[int, str]) -> Dataset:
    dataset = RussianTrafficSignBaseDataset(
        coco_ann_file=TRAIN_ANN_FILE,
        images_root=RTSD_IMG_ROOT,
        label_map_file=RTSD_LABELS_MAP_FILE,
    )
    dataset = LabelEnumApplier(dataset, label_enum_id2name=label_enum)
    dataset = limit_class_num_in_dataset(
        dataset, max_class_presence=MAX_AMOUNT_OF_SAMPLES_OF_ONE_CLASS
    )
    dataset = SquareCropReader(dataset)
    return dataset


def load_val_dataset(label_enum: Dict[int, str]) -> Dataset:
    dataset = RussianTrafficSignBaseDataset(
        coco_ann_file=VAL_ANN_FILE,
        images_root=RTSD_IMG_ROOT,
        label_map_file=RTSD_LABELS_MAP_FILE,
    )
    dataset = LabelEnumApplier(dataset, label_enum_id2name=label_enum)
    dataset = limit_class_num_in_dataset(
        dataset, max_class_presence=MAX_AMOUNT_OF_SAMPLES_OF_ONE_CLASS
    )
    dataset = SquareCropReader(dataset)
    return dataset


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_batch: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        if key == "datetime":
            continue
        collated_batch[key] = default_collate([sample[key] for sample in batch])
    collated_batch["datetime"] = [sample["datetime"] for sample in batch]
    return collated_batch


def init_clearml_task(task_name: str) -> Task:
    task = Task.init(
        project_name="TrafficSignProject",
        task_name=task_name,
        reuse_last_task_id=False,
    )
    return task


def get_experiment_name(base_experiment_name: str, experiments_root: str):
    existing_experiments = os.listdir(experiments_root)
    if base_experiment_name not in existing_experiments:
        return base_experiment_name
    namesakes_experiments = list(
        filter(
            lambda name: name.startswith(f"{base_experiment_name}"),
            existing_experiments,
        )
    )
    namesakes_numbers = [0]
    for namesake_name in namesakes_experiments:
        try:
            namesakes_number = int(namesake_name[len(base_experiment_name) :])
            namesakes_numbers.append(namesakes_number)
        except ValueError:
            continue
    new_exp_number = max(namesakes_numbers) + 1
    experiment_name = f"{base_experiment_name}{new_exp_number}"
    return experiment_name


@hydra.main(
    config_path=CFG_ROOT,
    config_name=CFG_NAME,
    version_base=None,
)
def train_static_gesture(cfg: TrafficSignTrainerConfig):
    experiment_name = get_experiment_name(
        "traffic_sign_recognition", TRAIN_RESULTS_ROOT
    )
    model_line_path: str = os.path.join(TRAIN_RESULTS_ROOT, experiment_name)
    model_line: ModelLine = ModelLine(model_line_path)
    lightning_classifier = TrafficSignsClassifier(cfg, results_location=model_line)
    task = init_clearml_task(experiment_name)
    train_dataset = load_train_dataset(cfg.model.label_enum)
    val_dataset = load_val_dataset(cfg.model.label_enum)
    augmentations = init_augmentations_from_config(augs_cfg=cfg.augs)
    train_dataset = TransformApplier(
        dataset=train_dataset, transformation=augmentations[DataSplit.TRAIN]
    )
    val_dataset = TransformApplier(
        dataset=val_dataset, transformation=augmentations[DataSplit.VAL]
    )
    print(f"TRAIN DS LEN = {len(train_dataset)}")
    print(f"VAL DS LEN = {len(val_dataset)}")
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
    )


if __name__ == "__main__":
    train_static_gesture()
