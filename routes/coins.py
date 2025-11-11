from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)


try:
    from smart_cache_system import coins_cache, raw_coins_cache
    SMART_CACHE_AVAILABLE = True
except ImportError:
    from debug_system.storage.cache_decorators import cache_coins, cache_raw_coins
    SMART_CACHE_AVAILABLE = False

coins_router = APIRouter(prefix="/api/coins", tags=["Coins"])

@coins_router.get("/list", summary="لیست نمادها")
@coins_cache
async def get_coins_list(
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    currency: str = Query("USD"),
    sort_by: str = Query("rank")
):
    """دریافت لیست نمادهای پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_coins_list(
            limit=limit, page=page, currency=currency, sort_by=sort_by
        )
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساده داده‌ها
        processed_coins = []
        for coin in raw_data.get('result', []):
            processed_coins.append({
                'id': coin.get('id'),
                'name': coin.get('name'),
                'symbol': coin.get('symbol'),
                'price': coin.get('price'),
                'price_change_24h': coin.get('priceChange1d'),
                'volume_24h': coin.get('volume'),
                'market_cap': coin.get('marketCap'),
                'rank': coin.get('rank'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_coins,
            'pagination': raw_data.get('meta', {}),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in coins list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@coins_router.get("/details/{coin_id}", summary="جزئیات نماد")
@coins_cache
async def get_coin_details(coin_id: str, currency: str = Query("USD")):
    """دریافت جزئیات پردازش شده یک نماد"""
    try:
        raw_data = coin_stats_manager.get_coin_details(coin_id, currency)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # 🔍 دیباگ: بررسی ساختار واقعی داده
        logger.info(f"🔍 Route received data: {raw_data}")
        
        # 🔧 اصلاح: استفاده از ساختار جدید manager
        # manager جدید ساختار {'status': 'success', 'data': {...}} برمی‌گرداند
        coin_data = raw_data.get('data', {})
        
        if not coin_data:
            # اگر داده خالی است، از raw_data مستقیم استفاده کن
            coin_data = raw_data
        
        # پردازش داده‌ها با ساختار جدید
        processed_data = {
            'id': coin_data.get('id'),
            'name': coin_data.get('name'),
            'symbol': coin_data.get('symbol'),
            'price': coin_data.get('price'),
            'price_change_24h': coin_data.get('price_change_24h', coin_data.get('priceChange1d')),
            'price_change_1h': coin_data.get('price_change_1h', coin_data.get('priceChange1h')),
            'price_change_1w': coin_data.get('price_change_1w', coin_data.get('priceChange1w')),
            'volume_24h': coin_data.get('volume_24h', coin_data.get('volume')),
            'market_cap': coin_data.get('market_cap', coin_data.get('marketCap')),
            'rank': coin_data.get('rank'),
            'website': coin_data.get('website', coin_data.get('websiteUrl')),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in coin details for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@coins_router.get("/charts/{coin_id}", summary="چارت نماد")
@coins_cache
async def get_coin_charts(coin_id: str, period: str = Query("1w")):
    """دریافت چارت پردازش شده نماد"""
    try:
        raw_data = coin_stats_manager.get_coin_charts(coin_id, period)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data': raw_data,
            'coin_id': coin_id,
            'period': period,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in coin charts for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@coins_router.get("/price/avg", summary="قیمت متوسط")
@coins_cache
async def get_coin_price_avg(
    coin_id: str = Query("bitcoin"),
    timestamp: str = Query("1636315200")
):
    """دریافت قیمت متوسط پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_coin_price_avg(coin_id, timestamp)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data': {
                'coin_id': coin_id,
                'timestamp': timestamp,
                'average_price': raw_data.get('price'),
                'currency': 'USD'
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in price avg for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
