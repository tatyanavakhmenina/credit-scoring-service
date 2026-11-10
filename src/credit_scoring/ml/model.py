import logging
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow import MlflowClient

from credit_scoring.config import Settings
from credit_scoring.schemas import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    alias: str
    version: str
    run_id: str
    threshold: float


class ScoringModel:
    def __init__(self, pipeline, info: ModelInfo) -> None:
        self._pipeline = pipeline
        self.info = info

    def predict_proba(self, features: dict) -> float:
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        return float(self._pipeline.predict_proba(frame)[0, 1])

    def decide(self, probability: float) -> str:
        return "reject" if probability >= self.info.threshold else "approve"


def load_model(settings: Settings) -> ScoringModel:
    """Блокирующая загрузка по alias: версия определяется только в Model Registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        settings.model_name, settings.model_alias
    )
    uri = f"models:/{settings.model_name}@{settings.model_alias}"
    pipeline = mlflow.sklearn.load_model(uri)
    info = ModelInfo(
        name=settings.model_name,
        alias=settings.model_alias,
        version=str(version.version),
        run_id=version.run_id,
        threshold=float(version.tags.get("threshold", 0.5)),
    )
    logger.info("model loaded", extra={"model_uri": uri, "version": info.version})
    return ScoringModel(pipeline, info)
