import os

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset

from training.data import DATASET_NAME, SOURCE_URL, TARGET

EXPERIMENT = "credit-default"
MODEL_NAME = "credit-default"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(EXPERIMENT)


def as_mlflow_dataset(frame: pd.DataFrame) -> PandasDataset:
    """source — откуда данные, digest — хеш содержимого: по нему видно смену данных."""
    return mlflow.data.from_pandas(
        frame, source=SOURCE_URL, name=DATASET_NAME, targets=TARGET
    )
