import asyncpg
from typing import Optional
from .config import settings


class DatabaseConnection:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=10,
                max_size=20,
                command_timeout=60
            )
    
    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    async def execute_query(self, query: str, *args):
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def execute_single(self, query: str, *args):
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)


db = DatabaseConnection()