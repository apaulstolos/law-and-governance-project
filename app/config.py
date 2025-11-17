from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_type: str = Field(env="OPENAI_API_TYPE", default="azure")
    azure_openai_endpoint: str = Field(env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(env="AZURE_OPENAI_API_VERSION", default="2024-02-01")
    azure_openai_embedding_deployment: str = Field(env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    azure_openai_chat_deployment: str = Field(env="AZURE_OPENAI_CHAT_DEPLOYMENT")

    api_host: str = Field(env="API_HOST", default="0.0.0.0")
    api_port: int = Field(env="API_PORT", default=8000)

    sqlite_path: Path = Field(env="SQLITE_PATH", default=Path("data/app.db"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
