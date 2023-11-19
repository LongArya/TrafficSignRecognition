import sys
import torch
from neptune.new.run import Run
from neptune.utils import stringify_unsupported
from torchmetrics.classification import BinaryPrecisionRecallCurve
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Union, Mapping, Any
from .classification_result_dataframe import ClassificationResultsDataframe


def generate_confusion_matrix_plot_from_classification_results(
    prediction_results: ClassificationResultsDataframe,
    plot_axis: Axes,
    labels: List[str],
) -> Axes:
    """Plots confusion matrix on given axis based on provided prediction results"""
    ground_true = prediction_results.ground_true.tolist()
    predictions = prediction_results.prediction.tolist()
    conf_mat: np.ndarray = confusion_matrix(
        y_true=ground_true,
        y_pred=predictions,
        labels=labels,
    )
    conf_mat_dataframe: pd.DataFrame = pd.DataFrame(
        data=conf_mat, index=labels, columns=labels
    )
    plot_axis = sns.heatmap(data=conf_mat_dataframe, ax=plot_axis, annot=True, fmt="g")
    return plot_axis
