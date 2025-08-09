from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from app.core.config import settings
from app.core.database import db
from app.api.endpoints import system, predictions, models, data, websocket, charts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    
    # Start background tasks
    from app.api.endpoints.websocket import monitor_predictions, monitor_system_status
    asyncio.create_task(monitor_predictions(db))
    asyncio.create_task(monitor_system_status())
    
    yield
    
    # Shutdown
    await db.disconnect()


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(websocket.router, tags=["websocket"])


@app.get("/")
async def root():
    return {
        "message": "BSE Predict Dashboard API",
        "version": settings.api_version,
        "docs": "/docs"
    }