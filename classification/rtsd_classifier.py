from datetime import datetime
from torchvision.models import resnet18, resnet34, resnet50
from neptune.types import File
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import pandas as pd
from PIL import Image
from typing import Dict, List, Iterable, Tuple, Callable, Optional
from collections import defaultdict
from .config import TrafficSignTrainerConfig, AugsConfig, TrainHyperparameters
from .focal_loss import FocalLoss
from .data_split import DataSplit
from .classification_result_dataframe import ClassificationResultsDataframe
from .model_line import ModelLine
from .metrics_utils import generate_confusion_matrix_plot_from_classification_results
from loggers.clearml_logger import ClearMLLogger
from dataset.rtsd import RussianTrafficSignBaseDataset
import torchvision.transforms as tf
from functools import partial
import albumentations as A
from albumentations import ImageOnlyTransform
import random


class WeatherAugmentApplier:
    def __init__(self, probabilty: float) -> None:
        rain = A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)
        snow = A.RandomSnow(
            brightness_coeff=0.5, snow_point_lower=0.1, snow_point_upper=0.3, p=1
        )
        fog = A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1)
        self.aug_probability = probabilty
        self.weather_transforms: List[ImageOnlyTransform] = [rain, snow, fog]

    def __call__(self, image: Image) -> Image:
        use_transform: bool = random.random() < self.aug_probability
        if not use_transform:
            return image

        transform = random.choice(self.weather_transforms)
        _transformed_image: np.ndarray = transform(image=np.array(image))["image"]
        transformed_image = Image.fromarray(_transformed_image)
        return transformed_image


def init_static_gesture_classifier(cfg: TrafficSignTrainerConfig) -> nn.Module:
    """Initializes neural network for static gesture classification, based on config values"""
    classes_num = len(cfg.model.label_enum)
    if cfg.model.architecture == "resnet18":
        model = resnet18(pretrained=cfg.model.use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, classes_num)
        return model
    if cfg.model.architecture == "resnet34":
        model = resnet34(pretrained=cfg.model.use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, classes_num)
        return model
    else:
        raise NotImplementedError(f"Unknown architecture: {cfg.model.architecture}")


def init_lr_scheduler(optimizer, cfg: TrafficSignTrainerConfig) -> Optional[nn.Module]:
    """Initialized learning rate scheduler based on config values"""
    if cfg.train_hyperparams.scheduler_type == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=cfg.train_hyperparams.lr_reduction_factor,
            patience=cfg.train_hyperparams.patience_epochs_num,
        )
        return scheduler
    else:
        raise NotImplementedError(
            f"Unknown scheduler type: {cfg.train_hyperparams.scheduler_type}"
        )


def init_classification_results_dataframe(
    label_enum: Dict[int, str]
) -> "ClassificationResultsDataframe":
    """Inits dataframe that is used for saving ground true and predicted values on classification dataset"""

    return pd.DataFrame(
        columns=["image_path", "ground_true", "prediction", "prediction_score"]
        + [label_name for label_id, label_name in label_enum.items()]
    )


def init_augmentations_from_config(
    augs_cfg: AugsConfig,
) -> Dict[DataSplit, Callable[[Image.Image], torch.Tensor]]:
    """Inits augmentations for each data_split based on config values"""
    resize = tf.Resize(augs_cfg.input_resolution)
    weather_augmentation = WeatherAugmentApplier(
        probabilty=augs_cfg.augmentation_probability
    )
    normalization = tf.Compose(
        [
            tf.ToTensor(),
            tf.Normalize(
                mean=augs_cfg.normalization_mean, std=augs_cfg.normalization_std
            ),
        ]
    )
    augmentation_transform = tf.Compose(
        [
            tf.ColorJitter(
                brightness=augs_cfg.brightness_factor,
                contrast=augs_cfg.contrast_factor,
                saturation=augs_cfg.saturation_contrast,
                hue=augs_cfg.hue_contrast,
            ),
            tf.RandomAffine(
                degrees=augs_cfg.rotation_range_angles_degrees,
                translate=augs_cfg.translation_range_imsize_fractions,
                scale=augs_cfg.scaling_range_factors,
                shear=augs_cfg.shear_x_axis_degrees_range,
            ),
        ]
    )
    val_aug = tf.Compose([resize, normalization])
    train_aug = tf.Compose(
        [
            resize,
            weather_augmentation,
            tf.RandomApply(
                transforms=[augmentation_transform], p=augs_cfg.augmentation_probability
            ),
            normalization,
        ]
    )
    return {DataSplit.TRAIN: train_aug, DataSplit.VAL: val_aug}


def init_loss_from_config(cfg: TrafficSignTrainerConfig) -> Optional[nn.Module]:
    """Initializes loss function based on config values"""
    if cfg.train_hyperparams.loss == "cross-entropy":
        return nn.CrossEntropyLoss()
    elif cfg.train_hyperparams.loss == "focal":
        return FocalLoss(
            class_num=len(cfg.model.label_enum),
            gamma=cfg.train_hyperparams.focal_loss_gamma,
        )
    else:
        raise NotImplementedError(
            f"Loss {cfg.train_hyperparams.loss} is not implemented"
        )


def label_hour(hour: str) -> str:
    if hour < 8:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "day"
    else:
        return "evening"


class TrafficSignsClassifier(pl.LightningModule):
    """Lightning module for training static gesture classification"""

    def __init__(
        self,
        cfg: TrafficSignTrainerConfig,
        results_location: ModelLine,
    ):
        super().__init__()
        self.cfg = cfg
        self.criterion = init_loss_from_config(cfg)
        self.model = init_static_gesture_classifier(self.cfg)
        self.results_dataframe_initalizer = partial(
            init_classification_results_dataframe, label_enum=self.cfg.model.label_enum
        )
        self.predictions_on_datasets: Dict[
            DataSplit, ClassificationResultsDataframe
        ] = defaultdict(self.results_dataframe_initalizer)
        self.results_location = results_location

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def get_gesture_prediction_for_single_input(
        self, single_input: torch.Tensor
    ) -> Tuple[str, float]:
        input_is_single_image: bool = single_input.ndim == 3 or (
            single_input.ndim == 4 and single_input.shape[0] == 1
        )
        if not input_is_single_image:
            raise ValueError("Should be used with single image only")
        network_input: torch.Tensor = (
            single_input
            if single_input.ndim == 4
            else torch.unsqueeze(single_input, dim=0)
        )
        network_input = network_input.to(self.device)
        logits = self.model(network_input).cpu()
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)[0]
        pred_class_index = torch.argmax(probs, dim=0, keepdim=True).item()
        prediction_probability: float = probs[pred_class_index].item()
        predicted_gesture = self.cfg.model.label_enum[pred_class_index]
        return predicted_gesture, prediction_probability

    def _append_predictions_to_split(
        self,
        gt_classes: torch.Tensor,
        logits: torch.Tensor,
        images_paths: List[str],
        split: DataSplit,
    ):
        """Saves predictions and corresponding ground true in the form of dataframe"""
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        pred_classes = torch.argmax(probs, dim=1, keepdim=True)
        predictions_scores = torch.take_along_dim(probs, pred_classes, dim=1)

        batch_predictions = []
        for path, gt_class, pred_class, pred_score, single_image_probabilities in zip(
            images_paths, gt_classes, pred_classes, predictions_scores, probs
        ):
            gt_label = self.cfg.model.label_enum[gt_class.item()]
            pred_label = self.cfg.model.label_enum[pred_class.item()]
            batch_predictions.append(
                [path, gt_label, pred_label, pred_score.item()]
                + [prob.item() for prob in single_image_probabilities]
            )
        batch_prediction_dataframe: ClassificationResultsDataframe = pd.DataFrame(
            batch_predictions,
            columns=self.predictions_on_datasets[split].columns,
        )
        self.predictions_on_datasets[split] = pd.concat(
            [self.predictions_on_datasets[split], batch_prediction_dataframe],
            ignore_index=True,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        gt_labels = batch["class"]
        images_paths = batch["image_path"]
        pred_labels = self.model(inputs)
        self._append_predictions_to_split(
            gt_classes=gt_labels,
            logits=pred_labels,
            images_paths=images_paths,
            split=DataSplit.TRAIN,
        )

        loss = self.criterion(pred_labels, gt_labels)
        batch_size = batch["image"].size(dim=0)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        gt_labels = batch["class"]
        images_paths = batch["image_path"]
        pred_labels = self.model(inputs)
        # save gt, preds to table
        self._append_predictions_to_split(
            gt_classes=gt_labels,
            logits=pred_labels,
            images_paths=images_paths,
            split=DataSplit.VAL,
        )

    def on_train_epoch_end(self) -> None:
        # save results
        save_path = os.path.join(
            self.results_location.train_predictions_root,
            f"{self.current_epoch:04d}.csv",
        )
        self.predictions_on_datasets[DataSplit.TRAIN].to_csv(save_path)
        # refresh results
        self.predictions_on_datasets[
            DataSplit.TRAIN
        ] = self.results_dataframe_initalizer()
        return super().on_train_epoch_end()

    def _log_conf_matrix_to_clearml(
        self,
        classification_results: ClassificationResultsDataframe,
        log_path: str,
    ) -> None:
        """Uploads confusion matrix to ClearML"""
        fig, ax = plt.subplots()
        ax = generate_confusion_matrix_plot_from_classification_results(
            classification_results, ax
        )
        plt.close(fig)
        raise NotImplementedError("TODO log conf mat to ClearML")

    def _get_classification_report_from_predictions(
        self, predictions: ClassificationResultsDataframe
    ) -> Dict:
        gt = predictions.ground_true.tolist()
        pred = predictions.prediction.tolist()
        labels_names = list(self.cfg.model.label_enum.values())
        val_classification_report = classification_report(
            y_true=gt, y_pred=pred, labels=labels_names, output_dict=True
        )
        return val_classification_report

    def _log_metrics_to_clearml(
        self,
        val_predictions: ClassificationResultsDataframe,
    ) -> None:
        labels_names = list(self.cfg.model.label_enum.values())
        val_classification_report = self._get_classification_report_from_predictions(
            val_predictions
        )
        if not isinstance(self.logger, ClearMLLogger):
            raise ValueError("Only ClearML logger is supported")

        self.logger.log_classification_report(
            sklearn_classifcation_report=val_classification_report,
            class_names=labels_names,
            step=self.current_epoch,
        )
        self.log(
            "val_weighted_F1", val_classification_report["weighted avg"]["f1-score"]
        )

    def _log_metrics_for_different_seasons_and_times(
        self, val_predictions: ClassificationResultsDataframe
    ) -> None:
        image_paths = val_predictions["image_path"]
        datetimes: List[datetime] = [
            RussianTrafficSignBaseDataset.extract_datetime_from_filename(path)
            for path in image_paths
        ]
        months_names = [date.strftime("%B") for date in datetimes]
        hours = [date.hour for date in datetimes]
        hours_labels = [label_hour(h) for h in hours]
        val_predictions["month_names"] = months_names
        val_predictions["daytime"] = hours_labels

        # log for every month
        month_grouping = val_predictions.groupby(by=["month_names"])
        logger: ClearMLLogger = self.logger
        for month, group in month_grouping:
            group_classification_report = (
                self._get_classification_report_from_predictions(group)
            )
            logger._clearml_logger.report_scalar(
                title="Month metrics",
                series=month,
                iteration=self.current_epoch,
                value=group_classification_report["weighted avg"]["f1-score"],
            )

        # log for every daytime
        daytime_grouping = val_predictions.groupby(by=["daytime"])
        for daytime, group in daytime_grouping:
            group_classification_report = (
                self._get_classification_report_from_predictions(group)
            )
            logger._clearml_logger.report_scalar(
                title="Daytime metrics",
                series=daytime,
                iteration=self.current_epoch,
                value=group_classification_report["weighted avg"]["f1-score"],
            )

        # log for every month and daytime
        daytime_grouping = val_predictions.groupby(by=["month_names", "daytime"])
        for (month, daytime), group in daytime_grouping:
            group_classification_report = (
                self._get_classification_report_from_predictions(group)
            )
            logger._clearml_logger.report_scalar(
                title="Month and Daytime metrics",
                series=f"{month}@{daytime}",
                iteration=self.current_epoch,
                value=group_classification_report["weighted avg"]["f1-score"],
            )

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_to_clearml(
            val_predictions=self.predictions_on_datasets[DataSplit.VAL]
        )
        self._log_metrics_for_different_seasons_and_times(
            val_predictions=self.predictions_on_datasets[DataSplit.VAL]
        )
        # save predictions locally
        save_path = os.path.join(
            self.results_location.val_predictions_root, f"{self.current_epoch:04d}.csv"
        )
        self.predictions_on_datasets[DataSplit.VAL].to_csv(save_path)
        # refresh predictions
        self.predictions_on_datasets[
            DataSplit.VAL
        ] = self.results_dataframe_initalizer()

        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.train_hyperparams.learinig_rate,
            momentum=self.cfg.train_hyperparams.momentun,
        )
        scheduler = init_lr_scheduler(optimizer=optimizer, cfg=self.cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
