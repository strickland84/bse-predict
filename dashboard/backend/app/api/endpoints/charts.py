from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta
from app.core.database import DatabaseConnection
from app.api.deps import get_db

router = APIRouter()


@router.get("/price-predictions")
async def get_price_predictions_chart_data(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    hours: int = Query(72, ge=24, le=168, description="Hours of data to fetch"),
    db: DatabaseConnection = Depends(get_db)
):
    """Get price data with predictions overlay for charting"""
    try:
        # Get price data (hourly candles)
        price_query = f"""
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_data
            WHERE timestamp > NOW() - INTERVAL '{hours} hours'
            {f"AND symbol = '{symbol}'" if symbol else ""}
            ORDER BY timestamp ASC
        """
        
        price_data = await db.execute_query(price_query)
        
        # Get predictions with outcomes
        predictions_query = f"""
            SELECT 
                p.timestamp as prediction_time,
                p.symbol,
                p.target_pct,
                p.prediction_class,
                p.confidence,
                p.probability,
                po.actual_outcome,
                po.target_hit_timestamp,
                po.max_favorable_move,
                po.max_adverse_move,
                po.completed_at
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE p.timestamp > NOW() - INTERVAL '{hours} hours'
            {f"AND p.symbol = '{symbol}'" if symbol else ""}
            ORDER BY p.timestamp ASC
        """
        
        predictions_data = await db.execute_query(predictions_query)
        
        # Organize data by symbol
        charts_data = {}
        
        # Process price data
        for row in price_data:
            sym = row['symbol']
            if sym not in charts_data:
                charts_data[sym] = {
                    'symbol': sym,
                    'price_data': [],
                    'predictions': {}
                }
            
            charts_data[sym]['price_data'].append({
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        # Process predictions
        for row in predictions_data:
            sym = row['symbol']
            if sym in charts_data:
                target_key = f"{float(row['target_pct'])*100:.0f}pct"
                if target_key not in charts_data[sym]['predictions']:
                    charts_data[sym]['predictions'][target_key] = []
                
                # Determine prediction status
                if row['actual_outcome'] is None:
                    status = 'pending'
                elif row['actual_outcome'] == row['prediction_class']:
                    status = 'correct'
                elif row['actual_outcome'] == -1:
                    status = 'expired'
                else:
                    status = 'incorrect'
                
                charts_data[sym]['predictions'][target_key].append({
                    'timestamp': row['prediction_time'].isoformat(),
                    'prediction': 'UP' if row['prediction_class'] == 1 else 'DOWN' if row['prediction_class'] == 0 else 'NEUTRAL',
                    'confidence': float(row['confidence']) * 100 if row['confidence'] else 0,
                    'probability': float(row['probability']) * 100 if row['probability'] else 0,
                    'status': status,
                    'actual_outcome': row['actual_outcome'],
                    'target_hit_time': row['target_hit_timestamp'].isoformat() if row['target_hit_timestamp'] else None,
                    'max_favorable': float(row['max_favorable_move']) if row['max_favorable_move'] else None,
                    'max_adverse': float(row['max_adverse_move']) if row['max_adverse_move'] else None
                })
        
        # Convert to list format
        result = list(charts_data.values())
        
        # Calculate statistics for each symbol
        for chart in result:
            total_predictions = 0
            correct_predictions = 0
            pending_predictions = 0
            
            for target_preds in chart['predictions'].values():
                for pred in target_preds:
                    total_predictions += 1
                    if pred['status'] == 'correct':
                        correct_predictions += 1
                    elif pred['status'] == 'pending':
                        pending_predictions += 1
            
            chart['statistics'] = {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'pending_predictions': pending_predictions,
                'accuracy': (correct_predictions / (total_predictions - pending_predictions) * 100) if (total_predictions - pending_predictions) > 0 else 0
            }
        
        return {
            'charts': result,
            'period_hours': hours,
            'generated_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-timeline")
async def get_performance_timeline(
    symbol: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=30),
    db: DatabaseConnection = Depends(get_db)
):
    """Get model performance over time for trend analysis"""
    try:
        query = f"""
            WITH daily_performance AS (
                SELECT 
                    DATE(p.timestamp) as date,
                    p.symbol,
                    p.target_pct,
                    COUNT(p.id) as total_predictions,
                    SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN po.actual_outcome IS NOT NULL AND po.actual_outcome != -1 THEN 1 ELSE 0 END) as completed
                FROM predictions p
                LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
                WHERE p.timestamp > NOW() - INTERVAL '{days} days'
                {f"AND p.symbol = '{symbol}'" if symbol else ""}
                GROUP BY DATE(p.timestamp), p.symbol, p.target_pct
            )
            SELECT 
                date,
                symbol,
                target_pct,
                total_predictions,
                correct,
                completed,
                CASE 
                    WHEN completed > 0 THEN ROUND(100.0 * correct / completed, 2)
                    ELSE 0 
                END as accuracy
            FROM daily_performance
            ORDER BY date ASC, symbol, target_pct
        """
        
        results = await db.execute_query(query)
        
        # Organize by symbol and target
        timeline_data = {}
        
        for row in results:
            sym = row['symbol']
            target = f"{float(row['target_pct'])*100:.0f}%"
            
            if sym not in timeline_data:
                timeline_data[sym] = {}
            if target not in timeline_data[sym]:
                timeline_data[sym][target] = []
            
            timeline_data[sym][target].append({
                'date': row['date'].isoformat(),
                'predictions': row['total_predictions'],
                'correct': row['correct'],
                'completed': row['completed'],
                'accuracy': float(row['accuracy'])
            })
        
        return {
            'timeline': timeline_data,
            'period_days': days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))