from typing import AsyncGenerator
from app.core.database import db


async def get_db() -> AsyncGenerator:
    try:
        yield db
    finally:
        pass