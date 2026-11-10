"""Создаёт бакет артефактов и запускает MLflow Tracking Server."""

import os

import boto3
from botocore.exceptions import ClientError

bucket = os.environ["MLFLOW_BUCKET"]
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
try:
    s3.head_bucket(Bucket=bucket)
except ClientError:
    s3.create_bucket(Bucket=bucket)

os.execvp(
    "mlflow",
    [
        "mlflow",
        "server",
        "--host=0.0.0.0",
        "--port=5000",
        "--workers=1",
        f"--backend-store-uri={os.environ['MLFLOW_BACKEND_URI']}",
        f"--artifacts-destination=s3://{bucket}",
        "--default-artifact-root=mlflow-artifacts:/",
        "--serve-artifacts",
        "--allowed-hosts=*",
    ],
)
