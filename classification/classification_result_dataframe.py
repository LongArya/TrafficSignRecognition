from dataclasses import dataclass
import pandas as pd


@dataclass
class ClassificationResultsDataframe(pd.DataFrame):
    @property
    def image_path(self) -> pd.Series:
        ...

    @property
    def ground_true(self) -> pd.Series:
        ...

    @property
    def prediction(self) -> pd.Series:
        ...

    @property
    def prediction_score(self) -> pd.Series:
        ...
