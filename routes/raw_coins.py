from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

raw_coins_router = APIRouter(prefix="/api/raw/coins", tags=["Raw Coins"])

@raw_coins_router.get("/list", summary="لیست خام نمادها")
async def get_raw_coins_list(
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    currency: str = Query("USD"),
    sort_by: str = Query("rank"),
    sort_dir: str = Query("asc")
):
    """دریافت لیست خام نمادها - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_coins_list(
            limit=limit, 
            page=page, 
            currency=currency, 
            sort_by=sort_by,
            sort_dir=sort_dir
        )
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'limit': limit,
                'page': page,
                'currency': currency,
                'sort_by': sort_by,
                'sort_dir': sort_dir
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw coins list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/details/{coin_id}", summary="جزئیات خام نماد")
async def get_raw_coin_details(coin_id: str, currency: str = Query("USD")):
    """دریافت جزئیات خام یک نماد - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_coin_details(coin_id, currency)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'coin_id': coin_id,
            'currency': currency,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw coin details for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/charts/{coin_id}", summary="چارت خام نماد")
async def get_raw_coin_charts(coin_id: str, period: str = Query("1w")):
    """دریافت چارت خام نماد - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_coin_charts(coin_id, period)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'coin_id': coin_id,
            'period': period,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw coin charts for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/multi-charts", summary="چارت خام چندنماد")
async def get_raw_multi_charts(
    coin_ids: str = Query(..., description="لیست coin_idها با کاما جدا شده"),
    period: str = Query("1w")
):
    """دریافت چارت خام چند نماد - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_coins_charts(coin_ids, period)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'coin_ids': coin_ids.split(','),
            'period': period,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw multi-charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/price/avg", summary="قیمت متوسط خام")
async def get_raw_coin_price_avg(
    coin_id: str = Query("bitcoin"),
    timestamp: str = Query("1636315200")
):
    """دریافت قیمت متوسط خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_coin_price_avg(coin_id, timestamp)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'coin_id': coin_id,
            'timestamp': timestamp,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw price avg for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/price/exchange", summary="قیمت صرافی خام")
async def get_raw_exchange_price(
    exchange: str = Query("Binance"),
    from_coin: str = Query("BTC"),
    to_coin: str = Query("ETH"),
    timestamp: str = Query("1636315200")
):
    """دریافت قیمت صرافی خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_exchange_price(exchange, from_coin, to_coin, timestamp)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'exchange': exchange,
            'from_coin': from_coin,
            'to_coin': to_coin,
            'timestamp': timestamp,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw exchange price: {e}")
        raise HTTPException(status_code=500, detail=str(e))
