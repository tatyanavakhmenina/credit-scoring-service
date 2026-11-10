from math import ceil

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    log_level: str = "INFO"
    git_sha: str = "local"

    database_url: str = "postgresql://postgres:postgres@localhost:5433/credit_scoring"
    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    mlflow_tracking_uri: str = "http://localhost:5001"

    model_name: str = "credit-default"
    model_alias: str = "champion"

    queue_name: str = "scoring.tasks"
    queue_max_length: int = 200  # 0 — без лимита (только для эксперимента)
    delivery_limit: int = 3
    publish_timeout_seconds: float = 1.0

    worker_concurrency: int = 4
    prefetch_count: int | None = None  # None — равен worker_concurrency, 0 — без лимита
    inference_delay_seconds: float = 0.2
    graceful_timeout_seconds: float = 30.0

    source_file: str = "data/credit_default.csv"
    source_batch_id: str = "batch-1"
    source_max_tasks: int = 30_000
    source_backoff_initial_seconds: float = 0.05
    source_backoff_max_seconds: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )

    @property
    def effective_prefetch(self) -> int:
        if self.prefetch_count is None:
            return self.worker_concurrency
        return self.prefetch_count

    @property
    def retry_after_seconds(self) -> int:
        """Время, за которое один воркер разберёт полную очередь (не меньше 1 с)."""
        throughput = self.worker_concurrency / max(self.inference_delay_seconds, 0.001)
        return max(1, ceil(self.queue_max_length / throughput))


settings = Settings()
