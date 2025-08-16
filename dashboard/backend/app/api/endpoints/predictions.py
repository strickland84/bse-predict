from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
from app.core.database import DatabaseConnection
from app.api.deps import get_db
from app.schemas.responses import PredictionData

router = APIRouter()


@router.get("/latest", response_model=List[PredictionData])
async def get_latest_predictions(db: DatabaseConnection = Depends(get_db)):
    try:
        query = """
            WITH latest_per_combo AS (
                SELECT 
                    symbol,
                    target_pct,
                    MAX(timestamp) as max_timestamp
                FROM predictions
                GROUP BY symbol, target_pct
            ),
            latest_predictions AS (
                SELECT 
                    p.symbol,
                    p.target_pct,
                    p.timestamp,
                    p.prediction_class,
                    p.confidence,
                    p.probability
                FROM predictions p
                INNER JOIN latest_per_combo lpc 
                    ON p.symbol = lpc.symbol 
                    AND p.target_pct = lpc.target_pct 
                    AND p.timestamp = lpc.max_timestamp
            ),
            pivoted AS (
                SELECT 
                    symbol,
                    MAX(timestamp) as timestamp,
                    STRING_AGG(CASE WHEN target_pct = 0.010 THEN prediction_class::text END, '') as prediction_1pct,
                    STRING_AGG(CASE WHEN target_pct = 0.020 THEN prediction_class::text END, '') as prediction_2pct,
                    STRING_AGG(CASE WHEN target_pct = 0.050 THEN prediction_class::text END, '') as prediction_5pct,
                    MAX(CASE WHEN target_pct = 0.010 THEN confidence END) as confidence_1pct,
                    MAX(CASE WHEN target_pct = 0.020 THEN confidence END) as confidence_2pct,
                    MAX(CASE WHEN target_pct = 0.050 THEN confidence END) as confidence_5pct
                FROM latest_predictions
                GROUP BY symbol
            )
            SELECT 
                p.symbol,
                p.timestamp,
                p.prediction_1pct,
                p.prediction_2pct,
                p.prediction_5pct,
                p.confidence_1pct,
                p.confidence_2pct,
                p.confidence_5pct,
                COALESCE((SELECT close FROM ohlcv_data WHERE symbol = p.symbol ORDER BY timestamp DESC LIMIT 1), 0) as current_price
            FROM pivoted p
            ORDER BY symbol
        """
        
        results = await db.execute_query(query)
        
        predictions = []
        if results:
            for row in results:
                # Convert prediction_class (0=DOWN, 1=UP) to frontend format (negative=DOWN, positive=UP, 0=NEUTRAL)
                def convert_prediction(val):
                    if val is None or val == '':
                        return 0  # No prediction = NEUTRAL
                    val = int(val) if isinstance(val, str) else val
                    if val == 1:
                        return 1  # UP
                    elif val == 0:
                        return -1  # DOWN
                    else:
                        return 0  # Unknown = NEUTRAL
                
                predictions.append(PredictionData(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    target_1pct=convert_prediction(row['prediction_1pct']),
                    target_2pct=convert_prediction(row['prediction_2pct']),
                    target_5pct=convert_prediction(row['prediction_5pct']),
                    confidence_1pct=float(row['confidence_1pct']) if row['confidence_1pct'] is not None else 0,
                    confidence_2pct=float(row['confidence_2pct']) if row['confidence_2pct'] is not None else 0,
                    confidence_5pct=float(row['confidence_5pct']) if row['confidence_5pct'] is not None else 0,
                    current_price=float(row['current_price']) if row['current_price'] else 0
                ))
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_prediction_history(
    symbol: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        base_query = """
            SELECT 
                symbol,
                timestamp,
                prediction_1pct,
                prediction_2pct,
                prediction_5pct,
                confidence_1pct,
                confidence_2pct,
                confidence_5pct,
                current_price,
                actual_price_1h,
                actual_move_1h
            FROM predictions
            WHERE timestamp > NOW() - INTERVAL '{} hours'
        """
        
        if symbol:
            base_query += f" AND symbol = '{symbol}'"
        
        base_query += " ORDER BY timestamp DESC"
        
        results = await db.execute_query(base_query.format(hours))
        
        history = []
        for row in results:
            history.append({
                "symbol": row['symbol'],
                "timestamp": row['timestamp'],
                "predictions": {
                    "1pct": row['prediction_1pct'],
                    "2pct": row['prediction_2pct'],
                    "5pct": row['prediction_5pct']
                },
                "confidence": {
                    "1pct": row['confidence_1pct'],
                    "2pct": row['confidence_2pct'],
                    "5pct": row['confidence_5pct']
                },
                "current_price": row['current_price'],
                "actual_price_1h": row['actual_price_1h'],
                "actual_move_1h": row['actual_move_1h']
            })
        
        return {"history": history, "count": len(history)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detailed")
async def get_predictions_detailed(
    symbol: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=10, le=100),
    outcome_filter: Optional[str] = Query(None, description="Filter by outcome: 'win', 'loss', 'pending'"),
    target_filter: Optional[float] = Query(None, description="Filter by target percentage: 1, 2, or 5"),
    min_confidence: Optional[float] = Query(None, ge=0, le=100, description="Minimum confidence threshold"),
    sort_by: str = Query("timestamp", description="Sort field: timestamp, confidence, target_pct"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Build WHERE conditions
        where_conditions = ["p.timestamp > NOW() - INTERVAL '30 days'"]
        
        if symbol:
            where_conditions.append(f"p.symbol = '{symbol}'")
        
        if target_filter:
            target_pct_value = target_filter / 100.0
            where_conditions.append(f"p.target_pct = {target_pct_value}")
        
        if min_confidence is not None:
            confidence_value = min_confidence / 100.0
            where_conditions.append(f"p.confidence >= {confidence_value}")
        
        if outcome_filter:
            if outcome_filter == 'win':
                where_conditions.append("po.actual_outcome = p.prediction_class")
            elif outcome_filter == 'loss':
                where_conditions.append("po.actual_outcome != p.prediction_class AND po.actual_outcome != -1 AND po.actual_outcome IS NOT NULL")
            elif outcome_filter == 'pending':
                where_conditions.append("(po.actual_outcome IS NULL OR po.actual_outcome = -1)")
        
        where_clause = " AND ".join(where_conditions)
        
        # Count total records for pagination
        count_query = f"""
            SELECT COUNT(*) as total
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE {where_clause}
        """
        
        count_result = await db.execute_query(count_query)
        total_count = count_result[0]['total'] if count_result else 0
        total_pages = (total_count + per_page - 1) // per_page
        
        # Determine sort column
        sort_column_map = {
            "timestamp": "p.timestamp",
            "confidence": "p.confidence",
            "target_pct": "p.target_pct",
            "symbol": "p.symbol"
        }
        sort_column = sort_column_map.get(sort_by, "p.timestamp")
        order = "ASC" if sort_order.lower() == "asc" else "DESC"
        
        # Get paginated data
        base_query = f"""
            SELECT 
                p.id,
                p.symbol,
                p.target_pct,
                p.timestamp as prediction_time,
                p.prediction_class,
                p.confidence,
                p.probability,
                po.actual_outcome,
                po.target_hit_timestamp,
                po.time_to_target_hours,
                po.max_favorable_move,
                po.max_adverse_move,
                po.completed_at
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE {where_clause}
            ORDER BY {sort_column} {order}
            LIMIT {per_page} OFFSET {offset}
        """
        
        results = await db.execute_query(base_query)
        
        detailed = []
        for row in results:
            detailed.append({
                "id": row['id'],
                "symbol": row['symbol'],
                "target_pct": float(row['target_pct']) * 100,
                "prediction_time": row['prediction_time'],
                "prediction": "UP" if row['prediction_class'] == 1 else "DOWN" if row['prediction_class'] == 0 else "NEUTRAL",
                "confidence": float(row['confidence']) * 100 if row['confidence'] else 0,
                "probability": float(row['probability']) * 100 if row['probability'] else 0,
                "actual_outcome": row['actual_outcome'],
                "target_hit_time": row['target_hit_timestamp'],
                "time_to_target": float(row['time_to_target_hours']) if row['time_to_target_hours'] else None,
                "max_favorable": float(row['max_favorable_move']) if row['max_favorable_move'] else None,
                "max_adverse": float(row['max_adverse_move']) if row['max_adverse_move'] else None,
                "completed_at": row['completed_at']
            })
        
        return {
            "predictions": detailed,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/accuracy")
async def get_prediction_accuracy(
    hours: int = Query(24, ge=1, le=168),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        query = """
            WITH accuracy_calc AS (
                SELECT 
                    symbol,
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN prediction_1pct = 1 AND actual_move_1h >= 1 THEN 1
                             WHEN prediction_1pct = -1 AND actual_move_1h <= -1 THEN 1
                             WHEN prediction_1pct = 0 AND ABS(actual_move_1h) < 1 THEN 1
                             ELSE 0 END)::float / NULLIF(COUNT(*), 0) as accuracy_1pct,
                    SUM(CASE WHEN prediction_2pct = 1 AND actual_move_1h >= 2 THEN 1
                             WHEN prediction_2pct = -1 AND actual_move_1h <= -2 THEN 1
                             WHEN prediction_2pct = 0 AND ABS(actual_move_1h) < 2 THEN 1
                             ELSE 0 END)::float / NULLIF(COUNT(*), 0) as accuracy_2pct,
                    SUM(CASE WHEN prediction_5pct = 1 AND actual_move_1h >= 5 THEN 1
                             WHEN prediction_5pct = -1 AND actual_move_1h <= -5 THEN 1
                             WHEN prediction_5pct = 0 AND ABS(actual_move_1h) < 5 THEN 1
                             ELSE 0 END)::float / NULLIF(COUNT(*), 0) as accuracy_5pct
                FROM predictions
                WHERE timestamp > NOW() - INTERVAL '{} hours'
                    AND actual_move_1h IS NOT NULL
                GROUP BY symbol
            )
            SELECT * FROM accuracy_calc
            ORDER BY symbol
        """
        
        results = await db.execute_query(query.format(hours))
        
        accuracy = []
        for row in results:
            accuracy.append({
                "symbol": row['symbol'],
                "total_predictions": row['total_predictions'],
                "accuracy": {
                    "1pct": round(row['accuracy_1pct'] * 100, 2) if row['accuracy_1pct'] else 0,
                    "2pct": round(row['accuracy_2pct'] * 100, 2) if row['accuracy_2pct'] else 0,
                    "5pct": round(row['accuracy_5pct'] * 100, 2) if row['accuracy_5pct'] else 0
                }
            })
        
        return {"accuracy": accuracy, "time_period_hours": hours}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_prediction_statistics(
    days: int = Query(7, ge=1, le=30),
    symbol: Optional[str] = Query(None),
    db: DatabaseConnection = Depends(get_db)
):
    """Get comprehensive prediction statistics including hit rates, MAE/MFE, and timing."""
    try:
        # Build WHERE clause for symbol filtering
        where_clause = f"p.timestamp >= NOW() - INTERVAL '{days} days'"
        if symbol:
            where_clause += f" AND p.symbol = '{symbol}'"
        
        # Overall hit rate statistics
        hit_rate_query = f"""
            SELECT 
                -- Overall statistics
                COUNT(p.id) as total_predictions,
                COUNT(po.id) as tracked_outcomes,
                SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct_predictions,
                SUM(CASE WHEN po.actual_outcome != p.prediction_class AND po.actual_outcome != -1 THEN 1 ELSE 0 END) as incorrect_predictions,
                SUM(CASE WHEN po.actual_outcome = -1 THEN 1 ELSE 0 END) as expired_predictions,
                
                -- Hit rates
                ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / 
                    NULLIF(COUNT(po.id) - SUM(CASE WHEN po.actual_outcome = -1 THEN 1 ELSE 0 END), 0), 2) as overall_hit_rate,
                
                -- Timing statistics for wins
                AVG(CASE WHEN po.actual_outcome = p.prediction_class THEN po.time_to_target_hours ELSE NULL END) as avg_time_to_win,
                MIN(CASE WHEN po.actual_outcome = p.prediction_class THEN po.time_to_target_hours ELSE NULL END) as min_time_to_win,
                MAX(CASE WHEN po.actual_outcome = p.prediction_class THEN po.time_to_target_hours ELSE NULL END) as max_time_to_win,
                
                -- Timing statistics for losses
                AVG(CASE WHEN po.actual_outcome != p.prediction_class AND po.actual_outcome != -1 
                    THEN po.time_to_target_hours ELSE NULL END) as avg_time_to_loss,
                
                -- MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)
                AVG(po.max_favorable_move) as avg_mfe,
                AVG(po.max_adverse_move) as avg_mae,
                AVG(CASE WHEN po.actual_outcome = p.prediction_class THEN po.max_favorable_move ELSE NULL END) as avg_mfe_wins,
                AVG(CASE WHEN po.actual_outcome != p.prediction_class AND po.actual_outcome != -1 
                    THEN po.max_adverse_move ELSE NULL END) as avg_mae_losses
                
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE {where_clause}
        """
        
        # Breakdown by symbol and target
        breakdown_query = f"""
            SELECT 
                p.symbol,
                p.target_pct,
                COUNT(p.id) as total_predictions,
                COUNT(po.id) as tracked_outcomes,
                SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN po.actual_outcome != p.prediction_class AND po.actual_outcome != -1 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN po.actual_outcome = -1 THEN 1 ELSE 0 END) as expired,
                
                ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / 
                    NULLIF(COUNT(po.id) - SUM(CASE WHEN po.actual_outcome = -1 THEN 1 ELSE 0 END), 0), 2) as hit_rate,
                
                AVG(CASE WHEN po.actual_outcome = p.prediction_class THEN po.time_to_target_hours ELSE NULL END) as avg_win_time,
                AVG(po.max_favorable_move) as avg_mfe,
                AVG(po.max_adverse_move) as avg_mae
                
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE {where_clause}
            GROUP BY p.symbol, p.target_pct
            ORDER BY p.symbol, p.target_pct
        """
        
        # Confidence cohort analysis
        confidence_cohort_query = f"""
            WITH confidence_buckets AS (
                SELECT 
                    p.id,
                    p.confidence,
                    po.actual_outcome,
                    p.prediction_class,
                    CASE 
                        WHEN p.confidence >= 0.9 THEN '90-100%'
                        WHEN p.confidence >= 0.8 THEN '80-90%'
                        WHEN p.confidence >= 0.7 THEN '70-80%'
                        WHEN p.confidence >= 0.6 THEN '60-70%'
                        WHEN p.confidence >= 0.5 THEN '50-60%'
                        ELSE 'Below 50%'
                    END as confidence_bucket
                FROM predictions p
                LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
                WHERE {where_clause}
                    AND po.id IS NOT NULL
                    AND po.actual_outcome != -1
            )
            SELECT 
                confidence_bucket,
                COUNT(*) as total,
                SUM(CASE WHEN actual_outcome = prediction_class THEN 1 ELSE 0 END) as correct,
                ROUND(100.0 * SUM(CASE WHEN actual_outcome = prediction_class THEN 1 ELSE 0 END) / 
                    NULLIF(COUNT(*), 0), 2) as accuracy,
                ROUND(AVG(confidence * 100), 2) as avg_confidence
            FROM confidence_buckets
            GROUP BY confidence_bucket
            ORDER BY 
                CASE confidence_bucket
                    WHEN '90-100%' THEN 1
                    WHEN '80-90%' THEN 2
                    WHEN '70-80%' THEN 3
                    WHEN '60-70%' THEN 4
                    WHEN '50-60%' THEN 5
                    ELSE 6
                END
        """
        
        # Execute all queries
        overall_stats = await db.execute_query(hit_rate_query)
        breakdown_stats = await db.execute_query(breakdown_query)
        confidence_cohorts = await db.execute_query(confidence_cohort_query)
        
        # Format overall statistics
        overall = overall_stats[0] if overall_stats else {}
        formatted_overall = {
            "total_predictions": overall.get('total_predictions', 0),
            "tracked_outcomes": overall.get('tracked_outcomes', 0),
            "wins": overall.get('correct_predictions', 0),
            "losses": overall.get('incorrect_predictions', 0),
            "expired": overall.get('expired_predictions', 0),
            "overall_hit_rate": float(overall.get('overall_hit_rate', 0)) if overall.get('overall_hit_rate') else 0,
            "timing": {
                "avg_time_to_win_hours": float(overall.get('avg_time_to_win', 0)) if overall.get('avg_time_to_win') else 0,
                "min_time_to_win_hours": float(overall.get('min_time_to_win', 0)) if overall.get('min_time_to_win') else 0,
                "max_time_to_win_hours": float(overall.get('max_time_to_win', 0)) if overall.get('max_time_to_win') else 0,
                "avg_time_to_loss_hours": float(overall.get('avg_time_to_loss', 0)) if overall.get('avg_time_to_loss') else 0
            },
            "excursions": {
                "avg_mfe": float(overall.get('avg_mfe', 0)) if overall.get('avg_mfe') else 0,
                "avg_mae": float(overall.get('avg_mae', 0)) if overall.get('avg_mae') else 0,
                "avg_mfe_wins": float(overall.get('avg_mfe_wins', 0)) if overall.get('avg_mfe_wins') else 0,
                "avg_mae_losses": float(overall.get('avg_mae_losses', 0)) if overall.get('avg_mae_losses') else 0
            }
        }
        
        # Format breakdown by symbol and target
        formatted_breakdown = []
        for row in breakdown_stats:
            formatted_breakdown.append({
                "symbol": row['symbol'],
                "target_pct": f"{float(row['target_pct'])*100:.1f}%",
                "total": row['total_predictions'],
                "wins": row['wins'],
                "losses": row['losses'],
                "expired": row['expired'],
                "hit_rate": float(row['hit_rate']) if row['hit_rate'] else 0,
                "avg_win_time": float(row['avg_win_time']) if row['avg_win_time'] else 0,
                "avg_mfe": float(row['avg_mfe']) if row['avg_mfe'] else 0,
                "avg_mae": float(row['avg_mae']) if row['avg_mae'] else 0
            })
        
        # Format confidence cohorts
        formatted_cohorts = []
        for row in confidence_cohorts:
            formatted_cohorts.append({
                "confidence_range": row['confidence_bucket'],
                "total_predictions": row['total'],
                "correct": row['correct'],
                "accuracy": float(row['accuracy']) if row['accuracy'] else 0,
                "avg_confidence": float(row['avg_confidence']) if row['avg_confidence'] else 0
            })
        
        return {
            "period_days": days,
            "overall": formatted_overall,
            "by_model": formatted_breakdown,
            "confidence_cohorts": formatted_cohorts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))