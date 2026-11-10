import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from credit_scoring.ml.features import add_credit_features
from training.data import TARGET, load_dataset
from training.tracking import MODEL_NAME, as_mlflow_dataset, setup_mlflow

SEED = 42
XGB_DEFAULT = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3}
XGB_TUNED = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}


def _steps(engineered: bool) -> list:
    return (
        [("features", FunctionTransformer(add_credit_features))] if engineered else []
    )


def _logreg(engineered: bool, **params) -> Pipeline:
    model = LogisticRegression(max_iter=2000, **params)
    return Pipeline(
        [*_steps(engineered), ("scale", StandardScaler()), ("model", model)]
    )


def _xgb(engineered: bool, **params) -> Pipeline:
    model = XGBClassifier(random_state=SEED, n_jobs=2, eval_metric="auc", **params)
    return Pipeline([*_steps(engineered), ("model", model)])


CANDIDATES = {
    "baseline-logreg": (_logreg, False, {}),
    "logreg-features-balanced": (_logreg, True, {"class_weight": "balanced"}),
    "xgboost-default": (_xgb, False, XGB_DEFAULT),
    "xgboost-tuned-features": (_xgb, True, XGB_TUNED),
}


def best_threshold(y_true: pd.Series, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[f1[:-1].argmax()])


def compute_metrics(y_true: pd.Series, proba, threshold: float) -> dict[str, float]:
    predicted = proba >= threshold
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, predicted),
        "precision": precision_score(y_true, predicted),
        "recall": recall_score(y_true, predicted),
        "brier": brier_score_loss(y_true, proba),
    }


def log_diagnostics(y_true: pd.Series, proba, threshold: float, pipeline) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        plots = {
            "roc_curve.png": lambda ax: RocCurveDisplay.from_predictions(
                y_true, proba, ax=ax
            ),
            "pr_curve.png": lambda ax: PrecisionRecallDisplay.from_predictions(
                y_true, proba, ax=ax
            ),
            "confusion_matrix.png": lambda ax: ConfusionMatrixDisplay.from_predictions(
                y_true, proba >= threshold, ax=ax
            ),
            "calibration.png": lambda ax: CalibrationDisplay.from_predictions(
                y_true, proba, n_bins=10, ax=ax
            ),
        }
        for name, plot in plots.items():
            figure, ax = plt.subplots(figsize=(5, 4))
            plot(ax)
            figure.tight_layout()
            figure.savefig(folder / name, dpi=120)
            plt.close(figure)
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            names = estimator.feature_names_in_
            importance = pd.Series(estimator.feature_importances_, index=names)
            importance.sort_values().to_csv(folder / "feature_importance.csv")
        mlflow.log_artifacts(tmp, artifact_path="diagnostics")


def main() -> None:
    setup_mlflow()
    frame = load_dataset()
    dataset = as_mlflow_dataset(frame)
    X, y = frame.drop(columns=[TARGET]), frame[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED
    )

    for run_name, (build, engineered, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=run_name):
            pipeline = build(engineered, **params)
            pipeline.fit(X_train, y_train)
            threshold = best_threshold(y_val, pipeline.predict_proba(X_val)[:, 1])
            mlflow.set_tag("stage", "training")
            mlflow.log_input(dataset, context="training")
            mlflow.log_params(
                {
                    "model_family": run_name.split("-")[0],
                    "engineered_features": engineered,
                    "seed": SEED,
                    "split": "60/20/20 stratified",
                    "threshold": round(threshold, 4),
                    **params,
                }
            )
            val_proba = pipeline.predict_proba(X_val)[:, 1]
            test_proba = pipeline.predict_proba(X_test)[:, 1]
            val_metrics = compute_metrics(y_val, val_proba, threshold)
            test_metrics = compute_metrics(y_test, test_proba, threshold)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            log_diagnostics(y_test, test_proba, threshold, pipeline)
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
                input_example=X_train.head(3),
                registered_model_name=MODEL_NAME,
            )
            client = mlflow.MlflowClient()
            version = info.registered_model_version
            client.set_model_version_tag(MODEL_NAME, version, "threshold", threshold)
            client.set_model_version_tag(
                MODEL_NAME, version, "val_roc_auc", val_metrics["roc_auc"]
            )
            print(
                f"{run_name}: v{version} val_roc_auc={val_metrics['roc_auc']:.4f} "
                f"test_roc_auc={test_metrics['roc_auc']:.4f} "
                f"test_f1={test_metrics['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
