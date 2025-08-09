from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Set
import asyncio
import json
from datetime import datetime
from app.core.database import DatabaseConnection
from app.api.deps import get_db

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


async def monitor_predictions(db: DatabaseConnection):
    """Monitor for new predictions and broadcast updates"""
    last_check = datetime.utcnow()
    
    while True:
        try:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            query = """
                SELECT 
                    symbol,
                    target_pct,
                    timestamp,
                    prediction_class,
                    confidence,
                    probability
                FROM predictions
                WHERE timestamp > $1
                ORDER BY timestamp DESC
                LIMIT 10
            """
            
            results = await db.execute_query(query, last_check)
            
            if results:
                for row in results:
                    message = {
                        "type": "prediction_update",
                        "data": {
                            "symbol": row['symbol'],
                            "target_pct": float(row['target_pct']) * 100,
                            "timestamp": row['timestamp'].isoformat(),
                            "prediction": "UP" if row['prediction_class'] == 1 else "DOWN" if row['prediction_class'] == 0 else "NEUTRAL",
                            "confidence": float(row['confidence']) * 100 if row['confidence'] else 0,
                            "probability": float(row['probability']) * 100 if row['probability'] else 0
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await manager.broadcast(json.dumps(message))
                
                last_check = datetime.utcnow()
                
        except Exception as e:
            print(f"Error in monitor_predictions: {e}")
            await asyncio.sleep(30)


async def monitor_system_status():
    """Monitor system status and broadcast updates"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            message = {
                "type": "system_status",
                "data": {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.broadcast(json.dumps(message))
            
        except Exception as e:
            print(f"Error in monitor_system_status: {e}")
            await asyncio.sleep(60)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "data": {"status": "connected"},
                "timestamp": datetime.utcnow().isoformat()
            }),
            websocket
        )
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "data": {},
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)