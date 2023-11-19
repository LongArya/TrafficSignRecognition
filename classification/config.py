from dataclasses import dataclass
from typing import Tuple, List, Dict


@dataclass
class AugsConfig:
    normalization_mean: List[float]
    normalization_std: List[float]
    input_resolution: Tuple[int, int]
    rotation_range_angles_degrees: Tuple[float, float]
    translation_range_imsize_fractions: Tuple[float, float]
    scaling_range_factors: Tuple[float, float]
    shear_x_axis_degrees_range: Tuple[float, float]
    brightness_factor: float
    contrast_factor: float
    saturation_contrast: float
    hue_contrast: float
    augmentation_probability: float


@dataclass
class TrainHyperparameters:
    device: str
    learinig_rate: float
    momentun: float
    scheduler_type: str
    # optimizer_type: str
    patience_epochs_num: int
    lr_reduction_factor: float
    loss: str
    focal_loss_gamma: float


@dataclass
class ModelConfig:
    architecture: str
    use_pretrained: bool
    label_enum: Dict[int, str]


@dataclass
class TrafficSignTrainerConfig:
    """Schema that defines config parameters for static gesture classifier training"""

    augs: AugsConfig
    train_hyperparams: TrainHyperparameters
    model: ModelConfig
