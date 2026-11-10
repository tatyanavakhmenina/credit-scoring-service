import argparse

import mlflow

from training.tracking import MODEL_NAME, setup_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Назначить alias версии модели")
    parser.add_argument(
        "--version", help="версия; без неё берётся лучшая по val_roc_auc"
    )
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    setup_mlflow()
    client = mlflow.MlflowClient()
    version = args.version
    if version is None:
        versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
        best = max(versions, key=lambda v: float(v.tags.get("val_roc_auc", 0)))
        version = best.version
    client.set_registered_model_alias(MODEL_NAME, args.alias, version)
    print(f"{MODEL_NAME}@{args.alias} -> v{version}")


if __name__ == "__main__":
    main()
