from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
from app.core.database import DatabaseConnection
from app.api.deps import get_db
from app.schemas.responses import DataHealth

router = APIRouter()


@router.get("/health", response_model=List[DataHealth])
async def get_data_health(db: DatabaseConnection = Depends(get_db)):
    try:
        # Use consistent gap detection logic with backend
        query = """
            WITH data_stats AS (
                SELECT 
                    symbol,
                    COUNT(*) as total_candles,
                    MAX(timestamp) as latest_candle,
                    MIN(timestamp) as oldest_candle
                FROM ohlcv_data
                WHERE timeframe = '1h'
                GROUP BY symbol
            ),
            gap_detection AS (
                SELECT 
                    symbol,
                    COUNT(*) as gaps_detected
                FROM (
                    SELECT 
                        symbol,
                        timestamp,
                        LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
                        timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as time_diff
                    FROM ohlcv_data
                    WHERE timeframe = '1h'
                        AND timestamp >= NOW() - INTERVAL '7 days'
                ) t
                WHERE time_diff > INTERVAL '1 hour 5 minutes'
                GROUP BY symbol
            )
            SELECT 
                ds.symbol,
                ds.total_candles,
                ds.latest_candle,
                ds.oldest_candle,
                COALESCE(gd.gaps_detected, 0) as gaps_detected,
                ROUND(
                    ds.total_candles::numeric / 
                    NULLIF(EXTRACT(EPOCH FROM (ds.latest_candle - ds.oldest_candle)) / 3600, 0) * 100, 
                    2
                ) as coverage_percentage
            FROM data_stats ds
            LEFT JOIN gap_detection gd ON ds.symbol = gd.symbol
            ORDER BY ds.symbol
        """
        
        results = await db.execute_query(query)
        
        health_data = []
        for row in results:
            coverage = min(row['coverage_percentage'] or 0, 100)
            
            # Determine status
            if row['gaps_detected'] > 10:
                status = "critical"
            elif row['gaps_detected'] > 0:
                status = "gaps_detected"
            else:
                status = "healthy"
            
            health_data.append(DataHealth(
                symbol=row['symbol'],
                total_candles=row['total_candles'],
                latest_candle=row['latest_candle'],
                oldest_candle=row['oldest_candle'],
                gaps_detected=row['gaps_detected'],
                coverage_percentage=coverage,
                status=status
            ))
        
        return health_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candles/latest")
async def get_latest_candles(
    symbol: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        base_query = """
            SELECT 
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_data
        """
        
        if symbol:
            base_query += f" WHERE symbol = '{symbol}'"
        
        base_query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        results = await db.execute_query(base_query)
        
        candles = []
        for row in results:
            candles.append({
                "symbol": row['symbol'],
                "timestamp": row['timestamp'],
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return {"candles": candles, "count": len(candles)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recovery-status")
async def get_recovery_status(db: DatabaseConnection = Depends(get_db)):
    try:
        # Check for recent gaps
        query = """
            WITH recent_gaps AS (
                SELECT 
                    symbol,
                    timestamp,
                    LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
                    timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as gap_duration
                FROM ohlcv_data
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            )
            SELECT 
                symbol,
                COUNT(*) FILTER (WHERE gap_duration > INTERVAL '1 hour 5 minutes') as gaps_found,
                MAX(gap_duration) as max_gap,
                AVG(gap_duration) FILTER (WHERE gap_duration > INTERVAL '1 hour 5 minutes') as avg_gap
            FROM recent_gaps
            GROUP BY symbol
        """
        
        results = await db.execute_query(query)
        
        recovery_status = []
        for row in results:
            recovery_status.append({
                "symbol": row['symbol'],
                "gaps_24h": row['gaps_found'] or 0,
                "max_gap_minutes": int(row['max_gap'].total_seconds() / 60) if row['max_gap'] else 0,
                "avg_gap_minutes": int(row['avg_gap'].total_seconds() / 60) if row['avg_gap'] else 0,
                "needs_recovery": (row['gaps_found'] or 0) > 0
            })
        
        return {
            "recovery_status": recovery_status,
            "total_gaps": sum(s['gaps_24h'] for s in recovery_status),
            "symbols_needing_recovery": [s['symbol'] for s in recovery_status if s['needs_recovery']]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gaps")
async def get_data_gaps(
    symbol: Optional[str] = Query(None),
    hours: int = Query(168, ge=1, le=720),
    db: DatabaseConnection = Depends(get_db)
):
    """Get detailed information about data gaps for debugging."""
    try:
        # Build WHERE clause
        where_clause = f"WHERE timeframe = '1h' AND timestamp >= NOW() - INTERVAL '{hours} hours'"
        if symbol:
            where_clause += f" AND symbol = '{symbol}'"
        
        # Find all gaps
        gaps_query = f"""
            WITH candle_data AS (
                SELECT 
                    symbol,
                    timestamp,
                    LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
                    timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as time_diff
                FROM ohlcv_data
                {where_clause}
                ORDER BY symbol, timestamp
            )
            SELECT 
                symbol,
                prev_timestamp as gap_start,
                timestamp as gap_end,
                EXTRACT(EPOCH FROM time_diff) / 3600 as gap_hours,
                time_diff as gap_duration
            FROM candle_data
            WHERE time_diff > INTERVAL '1 hour 5 minutes'
            ORDER BY symbol, prev_timestamp
        """
        
        gap_results = await db.execute_query(gaps_query)
        
        # Summary by symbol
        summary_query = f"""
            WITH candle_data AS (
                SELECT 
                    symbol,
                    timestamp,
                    LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
                    timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as time_diff
                FROM ohlcv_data
                {where_clause}
            )
            SELECT 
                symbol,
                COUNT(CASE WHEN time_diff > INTERVAL '1 hour 5 minutes' THEN 1 END) as gaps_count,
                COUNT(*) as total_candles,
                MIN(timestamp) as first_candle,
                MAX(timestamp) as last_candle,
                MAX(EXTRACT(EPOCH FROM time_diff) / 3600) as max_gap_hours
            FROM candle_data
            GROUP BY symbol
            ORDER BY symbol
        """
        
        summary_results = await db.execute_query(summary_query)
        
        # Format response
        gaps_by_symbol = {}
        for gap in gap_results:
            sym = gap['symbol']
            if sym not in gaps_by_symbol:
                gaps_by_symbol[sym] = []
            gaps_by_symbol[sym].append({
                'gap_start': gap['gap_start'],
                'gap_end': gap['gap_end'],
                'gap_hours': round(float(gap['gap_hours']), 2) if gap['gap_hours'] else 0,
                'gap_duration': str(gap['gap_duration'])
            })
        
        summary = []
        for row in summary_results:
            summary.append({
                'symbol': row['symbol'],
                'total_gaps': row['gaps_count'] or 0,
                'total_candles': row['total_candles'],
                'first_candle': row['first_candle'],
                'last_candle': row['last_candle'],
                'max_gap_hours': round(float(row['max_gap_hours']), 2) if row['max_gap_hours'] else 0,
                'gaps': gaps_by_symbol.get(row['symbol'], [])
            })
        
        return {
            'period_hours': hours,
            'summary': summary,
            'total_gaps_found': len(gap_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/futures")
async def get_futures_data(
    symbol: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        # Get futures data summary
        summary_query = """
            SELECT 
                symbol,
                COUNT(*) as total_records,
                MIN(timestamp) as oldest_record,
                MAX(timestamp) as latest_record,
                AVG(open_interest) as avg_open_interest,
                AVG(funding_rate) * 10000 as avg_funding_rate_bps,
                MIN(funding_rate) * 10000 as min_funding_rate_bps,
                MAX(funding_rate) * 10000 as max_funding_rate_bps,
                STDDEV(funding_rate) * 10000 as funding_rate_std_bps
            FROM futures_data
            WHERE timestamp > NOW() - INTERVAL '{} hours'
        """.format(hours)
        
        if symbol:
            summary_query += f" AND symbol = '{symbol}'"
        
        summary_query += " GROUP BY symbol ORDER BY symbol"
        
        summary_results = await db.execute_query(summary_query)
        
        # Get recent futures data
        recent_query = """
            SELECT 
                symbol,
                timestamp,
                open_interest,
                funding_rate * 10000 as funding_rate_bps,
                mark_price,
                top_trader_ratio,
                taker_buy_sell_ratio
            FROM futures_data
            WHERE timestamp > NOW() - INTERVAL '{} hours'
        """.format(hours)
        
        if symbol:
            recent_query += f" AND symbol = '{symbol}'"
        
        recent_query += " ORDER BY timestamp DESC LIMIT 500"
        
        recent_results = await db.execute_query(recent_query)
        
        # Process results
        summary = []
        for row in summary_results:
            summary.append({
                "symbol": row['symbol'],
                "total_records": row['total_records'],
                "oldest_record": row['oldest_record'],
                "latest_record": row['latest_record'],
                "avg_open_interest": round(float(row['avg_open_interest']), 2) if row['avg_open_interest'] else 0,
                "avg_funding_rate_bps": round(float(row['avg_funding_rate_bps']), 4) if row['avg_funding_rate_bps'] else 0,
                "min_funding_rate_bps": round(float(row['min_funding_rate_bps']), 4) if row['min_funding_rate_bps'] else 0,
                "max_funding_rate_bps": round(float(row['max_funding_rate_bps']), 4) if row['max_funding_rate_bps'] else 0,
                "funding_volatility_bps": round(float(row['funding_rate_std_bps']), 4) if row['funding_rate_std_bps'] else 0
            })
        
        recent_data = []
        for row in recent_results:
            recent_data.append({
                "symbol": row['symbol'],
                "timestamp": row['timestamp'],
                "open_interest": round(float(row['open_interest']), 2) if row['open_interest'] else 0,
                "funding_rate_bps": round(float(row['funding_rate_bps']), 4) if row['funding_rate_bps'] else 0,
                "mark_price": round(float(row['mark_price']), 2) if row['mark_price'] else 0,
                "top_trader_ratio": round(float(row['top_trader_ratio']), 2) if row['top_trader_ratio'] else 0,
                "taker_buy_sell_ratio": round(float(row['taker_buy_sell_ratio']), 2) if row['taker_buy_sell_ratio'] else 0
            })
        
        return {
            "summary": summary,
            "recent_data": recent_data,
            "hours_analyzed": hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integrity-detailed")
async def get_data_integrity_detailed(
    symbol: Optional[str] = Query(None),
    db: DatabaseConnection = Depends(get_db)
):
    try:
        # Get detailed gap analysis
        gap_query = """
            WITH gap_analysis AS (
                SELECT 
                    symbol,
                    timestamp,
                    LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
                    timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as gap_duration,
                    close as price
                FROM ohlcv_data
                WHERE timestamp > NOW() - INTERVAL '30 days'
            )
            SELECT 
                symbol,
                timestamp as gap_start,
                prev_timestamp as gap_end,
                gap_duration,
                price,
                EXTRACT(EPOCH FROM gap_duration) / 3600 as gap_hours
            FROM gap_analysis
            WHERE gap_duration > INTERVAL '1 hour 5 minutes'
        """
        
        if symbol:
            gap_query += f" AND symbol = '{symbol}'"
        
        gap_query += " ORDER BY timestamp DESC LIMIT 100"
        
        gap_results = await db.execute_query(gap_query)
        
        # Get hourly statistics
        stats_query = """
            SELECT 
                symbol,
                DATE_TRUNC('day', timestamp) as day,
                COUNT(*) as candles_count,
                MIN(low) as day_low,
                MAX(high) as day_high,
                AVG(volume) as avg_volume,
                STDDEV(close) as price_volatility
            FROM ohlcv_data
            WHERE timestamp > NOW() - INTERVAL '30 days'
        """
        
        if symbol:
            stats_query += f" AND symbol = '{symbol}'"
        
        stats_query += " GROUP BY symbol, day ORDER BY day DESC"
        
        stats_results = await db.execute_query(stats_query)
        
        # Process results
        gaps = []
        for row in gap_results:
            gaps.append({
                "symbol": row['symbol'],
                "gap_start": row['gap_start'],
                "gap_end": row['gap_end'],
                "gap_hours": round(row['gap_hours'], 2) if row['gap_hours'] else 0,
                "price_at_gap": float(row['price']) if row['price'] else 0
            })
        
        daily_stats = []
        for row in stats_results:
            daily_stats.append({
                "symbol": row['symbol'],
                "date": row['day'],
                "candles_count": row['candles_count'],
                "expected_candles": 24,  # For hourly data
                "completeness": round((row['candles_count'] / 24) * 100, 2),
                "day_low": float(row['day_low']) if row['day_low'] else 0,
                "day_high": float(row['day_high']) if row['day_high'] else 0,
                "avg_volume": float(row['avg_volume']) if row['avg_volume'] else 0,
                "price_volatility": float(row['price_volatility']) if row['price_volatility'] else 0
            })
        
        return {
            "gaps": gaps,
            "daily_stats": daily_stats,
            "summary": {
                "total_gaps": len(gaps),
                "avg_gap_hours": round(sum(g['gap_hours'] for g in gaps) / len(gaps), 2) if gaps else 0,
                "data_completeness": round(sum(s['completeness'] for s in daily_stats) / len(daily_stats), 2) if daily_stats else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))