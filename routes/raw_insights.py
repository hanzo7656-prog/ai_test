from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

raw_insights_router = APIRouter(prefix="/api/raw/insights", tags=["Raw Insights"])

@raw_insights_router.get("/btc-dominance", summary="دامیننس بیت‌کوین خام")
async def get_raw_btc_dominance(period_type: str = Query("all")):
    """دریافت دامیننس بیت‌کوین خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_btc_dominance(period_type)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'period_type': period_type,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw BTC dominance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_insights_router.get("/fear-greed", summary="شاخص ترس و طمع خام")
async def get_raw_fear_greed():
    """دریافت شاخص ترس و طمع خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_fear_greed()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw fear-greed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_insights_router.get("/fear-greed/chart", summary="چارت ترس و طمع خام")
async def get_raw_fear_greed_chart():
    """دریافت چارت ترس و طمع خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_fear_greed_chart()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw fear-greed chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_insights_router.get("/rainbow-chart/{coin_id}", summary="چارت رنگین‌کمان خام")
async def get_raw_rainbow_chart(coin_id: str):
    """دریافت چارت رنگین‌کمان خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_rainbow_chart(coin_id)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'coin_id': coin_id,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw rainbow chart for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
