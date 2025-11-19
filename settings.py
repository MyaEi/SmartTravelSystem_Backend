# SmartTravelSystem_Backend/settings.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # repo root (where app.py lives)

class Settings(BaseSettings):
    # API keys
    OPENTRIPMAP_KEY: str = ""
    PREDICTHQ_KEY: str = ""
    CALENDARIFIC_KEY: str = ""
    UNSPLASH_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()


 
