from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime
from app.core.database import DatabaseConnection
from app.api.deps import get_db
from app.schemas.responses import SystemStatus

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@router.get("/status", response_model=SystemStatus)
async def get_system_status(db: DatabaseConnection = Depends(get_db)):
    try:
        # Check database connection
        db_healthy = False
        try:
            await db.execute_single("SELECT 1")
            db_healthy = True
        except:
            pass
        
        # Services status (simplified - just check if we can reach the database)
        services = {
            "database": "running" if db_healthy else "stopped",
            "backend": "running"  # We know backend is running if this endpoint responds
        }
        
        # Get latest timestamps
        last_data_fetch = None
        last_prediction = None
        last_training = None
        
        try:
            result = await db.execute_single("""
                SELECT MAX(timestamp) as latest 
                FROM ohlcv_data
            """)
            if result:
                last_data_fetch = result['latest']
            
            result = await db.execute_single("""
                SELECT MAX(timestamp) as latest 
                FROM predictions
            """)
            if result:
                last_prediction = result['latest']
            
            result = await db.execute_single("""
                SELECT MAX(trained_at) as latest 
                FROM model_training_history
            """)
            if result:
                last_training = result['latest']
        except:
            pass
        
        # Determine overall status
        overall_status = "healthy"
        if not db_healthy:
            overall_status = "error"
        
        return SystemStatus(
            status=overall_status,
            database=db_healthy,
            services=services,
            last_data_fetch=last_data_fetch,
            last_prediction=last_prediction,
            last_model_training=last_training
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


