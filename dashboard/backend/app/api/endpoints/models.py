from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import json
from app.core.database import DatabaseConnection
from app.api.deps import get_db
from app.schemas.responses import ModelPerformance

router = APIRouter()


@router.get("/performance", response_model=List[ModelPerformance])
async def get_model_performance(db: DatabaseConnection = Depends(get_db)):
    try:
        query = """
            SELECT 
                symbol,
                target_pct as target_percentage,
                final_accuracy as accuracy,
                precision as precision_score,
                recall as recall_score,
                f1_score,
                trained_at,
                training_samples
            FROM model_training_history
            WHERE (symbol, target_pct, trained_at) IN (
                SELECT symbol, target_pct, MAX(trained_at)
                FROM model_training_history
                GROUP BY symbol, target_pct
            )
            ORDER BY symbol, target_pct
        """
        
        results = await db.execute_query(query)
        
        performance = []
        if results:
            for row in results:
                performance.append(ModelPerformance(
                    symbol=row['symbol'],
                    target=f"{float(row['target_percentage'])*100:.0f}%",
                    accuracy=round(row['accuracy'] * 100, 2) if row['accuracy'] else 0,
                    precision=round(row['precision_score'] * 100, 2) if row['precision_score'] else 0,
                    recall=round(row['recall_score'] * 100, 2) if row['recall_score'] else 0,
                    f1_score=round(row['f1_score'] * 100, 2) if row['f1_score'] else 0,
                    last_trained=row['trained_at'],
                    training_samples=row['training_samples']
                ))
        
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-status")
async def get_training_status(db: DatabaseConnection = Depends(get_db)):
    try:
        query = """
            SELECT 
                symbol,
                target_pct as target_percentage,
                MAX(trained_at) as last_trained,
                COUNT(*) as total_trainings,
                AVG(final_accuracy) as avg_accuracy
            FROM model_training_history
            WHERE trained_at > NOW() - INTERVAL '7 days'
            GROUP BY symbol, target_pct
            ORDER BY symbol, target_pct
        """
        
        results = await db.execute_query(query)
        
        status = {}
        for row in results:
            if row['symbol'] not in status:
                status[row['symbol']] = {}
            
            status[row['symbol']][f"{row['target_percentage']}%"] = {
                "last_trained": row['last_trained'],
                "total_trainings": row['total_trainings'],
                "avg_accuracy": round(row['avg_accuracy'] * 100, 2)
            }
        
        return {"training_status": status}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance(db: DatabaseConnection = Depends(get_db)):
    try:
        # Since we're using RandomForest, we would need to load the models
        # For now, return mock data or pre-computed importance
        feature_importance = {
            "BTC/USDT": {
                "1%": {
                    "rsi": 0.15,
                    "macd_signal": 0.12,
                    "bb_position": 0.10,
                    "volume_ratio": 0.09,
                    "price_change_1h": 0.08
                },
                "2%": {
                    "rsi": 0.14,
                    "macd_signal": 0.13,
                    "bb_position": 0.11,
                    "volume_ratio": 0.10,
                    "price_change_2h": 0.09
                },
                "5%": {
                    "rsi": 0.16,
                    "macd_signal": 0.14,
                    "bb_position": 0.12,
                    "volume_ratio": 0.11,
                    "price_change_4h": 0.10
                }
            },
            "ETH/USDT": {
                "1%": {
                    "rsi": 0.14,
                    "macd_signal": 0.13,
                    "bb_position": 0.11,
                    "volume_ratio": 0.10,
                    "price_change_1h": 0.09
                },
                "2%": {
                    "rsi": 0.15,
                    "macd_signal": 0.12,
                    "bb_position": 0.11,
                    "volume_ratio": 0.10,
                    "price_change_2h": 0.08
                },
                "5%": {
                    "rsi": 0.17,
                    "macd_signal": 0.13,
                    "bb_position": 0.11,
                    "volume_ratio": 0.10,
                    "price_change_4h": 0.09
                }
            },
            "SOL/USDT": {
                "1%": {
                    "rsi": 0.13,
                    "macd_signal": 0.12,
                    "bb_position": 0.12,
                    "volume_ratio": 0.11,
                    "price_change_1h": 0.10
                },
                "2%": {
                    "rsi": 0.14,
                    "macd_signal": 0.13,
                    "bb_position": 0.12,
                    "volume_ratio": 0.10,
                    "price_change_2h": 0.09
                },
                "5%": {
                    "rsi": 0.15,
                    "macd_signal": 0.14,
                    "bb_position": 0.13,
                    "volume_ratio": 0.11,
                    "price_change_4h": 0.10
                }
            }
        }
        
        return {"feature_importance": feature_importance}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-history-detailed")
async def get_training_history_detailed(
    symbol: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=5, le=50),
    target_filter: Optional[float] = Query(None, description="Filter by target percentage: 1, 2, or 5"),
    min_accuracy: Optional[float] = Query(None, ge=0, le=100, description="Minimum accuracy threshold"),
    days_ago: Optional[int] = Query(None, ge=1, le=365, description="Filter by training date within N days"),
    sort_by: str = Query("trained_at", description="Sort field: trained_at, final_accuracy, f1_score, training_samples"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Build WHERE conditions
        where_conditions = []
        
        if symbol:
            where_conditions.append(f"symbol = '{symbol}'")
        
        if target_filter:
            target_pct_value = target_filter / 100.0
            where_conditions.append(f"target_pct = {target_pct_value}")
        
        if min_accuracy is not None:
            accuracy_value = min_accuracy / 100.0
            where_conditions.append(f"final_accuracy >= {accuracy_value}")
        
        if days_ago:
            where_conditions.append(f"trained_at > NOW() - INTERVAL '{days_ago} days'")
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Count total records for pagination
        count_query = f"SELECT COUNT(*) as total FROM model_training_history{where_clause}"
        
        count_result = await db.execute_query(count_query)
        total_count = count_result[0]['total'] if count_result else 0
        total_pages = (total_count + per_page - 1) // per_page
        
        # Determine sort column
        sort_column_map = {
            "trained_at": "trained_at",
            "final_accuracy": "final_accuracy",
            "f1_score": "f1_score",
            "training_samples": "training_samples",
            "cv_accuracy": "cv_accuracy",
            "precision": "precision",
            "recall": "recall"
        }
        sort_column = sort_column_map.get(sort_by, "trained_at")
        order = "ASC" if sort_order.lower() == "asc" else "DESC"
        
        # Get paginated data
        base_query = f"""
            SELECT 
                id,
                symbol,
                target_pct,
                trained_at,
                model_filename,
                training_samples,
                features_count,
                date_range_start,
                date_range_end,
                price_range_min,
                price_range_max,
                target_distribution,
                cv_accuracy,
                cv_std,
                final_accuracy,
                precision,
                recall,
                f1_score,
                top_features,
                model_config
            FROM model_training_history
            {where_clause}
            ORDER BY {sort_column} {order}
            LIMIT {per_page} OFFSET {offset}
        """
        
        results = await db.execute_query(base_query)
        
        history = []
        for row in results:
            history.append({
                "id": row['id'],
                "symbol": row['symbol'],
                "target_pct": float(row['target_pct']) * 100,
                "trained_at": row['trained_at'],
                "model_filename": row['model_filename'],
                "training_samples": row['training_samples'],
                "features_count": row['features_count'],
                "date_range": {
                    "start": row['date_range_start'],
                    "end": row['date_range_end']
                },
                "price_range": {
                    "min": float(row['price_range_min']),
                    "max": float(row['price_range_max'])
                },
                "target_distribution": json.loads(row['target_distribution']) if row['target_distribution'] and isinstance(row['target_distribution'], str) else row['target_distribution'] if row['target_distribution'] else {},
                "metrics": {
                    "cv_accuracy": float(row['cv_accuracy']) * 100,
                    "cv_std": float(row['cv_std']) * 100,
                    "final_accuracy": float(row['final_accuracy']) * 100,
                    "precision": float(row['precision']) * 100,
                    "recall": float(row['recall']) * 100,
                    "f1_score": float(row['f1_score']) * 100
                },
                "top_features": json.loads(row['top_features']) if row['top_features'] and isinstance(row['top_features'], str) else row['top_features'] if row['top_features'] else [],
                "model_config": json.loads(row['model_config']) if row['model_config'] and isinstance(row['model_config'], str) else row['model_config'] if row['model_config'] else {}
            })
        
        return {
            "training_history": history,
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