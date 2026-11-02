from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/credit_scoring"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
