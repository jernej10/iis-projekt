import pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DAGSHUB_TOKEN: str
    MONGO_URI: str
    GMAIL_PASSWORD: str
    GMAIL: str

    __project_root = pathlib.Path(__file__).resolve().parent.parent

    model_config = SettingsConfigDict(env_file=f"{__project_root}/.env")


settings = Settings()