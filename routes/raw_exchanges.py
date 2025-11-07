from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

raw_exchanges_router = APIRouter(prefix="/api/raw/exchanges", tags=["Raw Exchanges"])

@raw_exchanges_router.get("/list", summary="لیست خام صرافی‌ها")
async def get_raw_exchanges_list():
    """دریافت لیست خام صرافی‌ها - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_tickers_exchanges()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw exchanges list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/markets", summary="مارکت‌های خام")
async def get_raw_markets():
    """دریافت مارکت‌های خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_markets()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/tickers-markets", summary="مارکت‌های تیکر خام")
async def get_raw_tickers_markets():
    """دریافت مارکت‌های تیکر خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_tickers_markets()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw tickers markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/fiats", summary="ارزهای فیات خام")
async def get_raw_fiats():
    """دریافت ارزهای فیات خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_fiats()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw fiats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/currencies", summary="ارزهای خام")
async def get_raw_currencies():
    """دریافت ارزهای خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_currencies()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw currencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
