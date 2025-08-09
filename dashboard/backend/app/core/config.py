import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bse_predict")
    api_title: str = "BSE Predict Dashboard API"
    api_version: str = "1.0.0"
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"


settings = Settings()