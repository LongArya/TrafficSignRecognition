from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from clearml import Logger as ClearMLLogger
from typing import Dict, List, Any


def log_classififcation_report_to_clearml(
    clearml_logger: Logger,
    classification_report: Dict,
    class_names: List[str],
    iteration: int,
) -> None:
    report_metrics_names: List[str] = ["f1-score", "precision", "recall", "support"]
    for metric_name in report_metrics_names:
        title = "Per class " + metric_name
        for class_name in class_names:
            logged_value: float = classification_report[class_name][metric_name]
            clearml_logger.report_scalar(
                title=title,
                series=class_name,
                iteration=iteration,
                value=logged_value,
            )

    # log aggregated metrics
    aggregated_metrics_keys: List[str] = list(
        set(classification_report.keys()) - set(class_names) - set(["accuracy"])
    )
    for aggregated_metrics_key in aggregated_metrics_keys:
        aggregated_metrics = classification_report[aggregated_metrics_key]
        for series_name, series_value in aggregated_metrics.items():
            clearml_logger.report_scalar(
                title=aggregated_metrics_key,
                series=series_name,
                value=series_value,
                iteration=iteration,
            )

    # log accuracy
    clearml_logger.report_scalar(
        title="accuracy",
        series="accuracy",
        iteration=iteration,
        value=classification_report["accuracy"],
    )


class ClearMLLogger(Logger):
    def __init__(self, clearml_logger: ClearMLLogger):
        self._clearml_logger = clearml_logger

    @property
    def name(self):
        return "ClearMLLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def log_classification_report(
        self, sklearn_classifcation_report: Dict, class_names: List[str], step: int
    ) -> None:
        log_classififcation_report_to_clearml(
            clearml_logger=self._clearml_logger,
            classification_report=sklearn_classifcation_report,
            class_names=class_names,
            iteration=step,
        )

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
