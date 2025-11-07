from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

coins_router = APIRouter(prefix="/api/coins", tags=["Coins"])

@coins_router.get("/list", summary="لیست نمادها")
async def get_coins_list(
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    currency: str = Query("USD"),
    sort_by: str = Query("rank")
):
    """دریافت لیست نمادهای پردازش شده"""
    try:
        processed_data = coin_stats_manager.get_coins_list_processed(
            limit=limit, page=page, currency=currency, sort_by=sort_by
        )
        
        if "error" in processed_data:
            raise HTTPException(status_code=500, detail=processed_data["error"])
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in coins list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@coins_router.get("/details/{coin_id}", summary="جزئیات نماد")
async def get_coin_details(coin_id: str, currency: str = Query("USD")):
    """دریافت جزئیات پردازش شده یک نماد"""
    try:
        raw_data = coin_stats_manager.get_coin_details(coin_id, currency)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌ها
        processed_data = {
            'id': raw_data.get('id'),
            'name': raw_data.get('name'),
            'symbol': raw_data.get('symbol'),
            'price': raw_data.get('price'),
            'price_change_24h': raw_data.get('priceChange1d'),
            'price_change_1h': raw_data.get('priceChange1h'),
            'price_change_1w': raw_data.get('priceChange1w'),
            'volume_24h': raw_data.get('volume'),
            'market_cap': raw_data.get('marketCap'),
            'rank': raw_data.get('rank'),
            'website': raw_data.get('websiteUrl'),
            'description': raw_data.get('description'),
            'links': raw_data.get('links', []),
            'analysis': {
                'trend': _analyze_trend(raw_data),
                'signal': _generate_signal(raw_data),
                'confidence': _calculate_confidence(raw_data)
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in coin details for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@coins_router.get("/charts/{coin_id}", summary="چارت نماد")
async def get_coin_charts(coin_id: str, period: str = Query("all")):
    """دریافت چارت پردازش شده نماد"""
    try:
        raw_data = coin_stats_manager.get_coin_charts(coin_id, period)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌های چارت
        processed_charts = {
            'coin_id': coin_id,
            'period': period,
            'prices': raw_data.get('prices', []),
            'analysis': {
                'trend': _analyze_chart_trend(raw_data.get('prices', [])),
                'volatility': _calculate_volatility(raw_data.get('prices', [])),
                'support_resistance': _find_support_resistance(raw_data.get('prices', []))
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_charts,
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in coin charts for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@coins_router.get("/multi-charts", summary="چارت چندنماد")
async def get_multi_charts(
    coin_ids: str = Query(..., description="لیست coin_idها با کاما جدا شده"),
    period: str = Query("all")
):
    """دریافت چارت پردازش شده چند نماد"""
    try:
        raw_data = coin_stats_manager.get_coins_charts(coin_ids, period)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data': raw_data,
            'coin_ids': coin_ids.split(','),
            'period': period,
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in multi-charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@coins_router.get("/price/avg", summary="قیمت متوسط")
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
                'currency': 'USD',
                'calculated_at': datetime.now().isoformat()
            },
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in price avg for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# توابع کمکی پردازش
def _analyze_trend(coin_data: Dict) -> str:
    """تحلیل روند نماد"""
    change = coin_data.get('priceChange1d', 0)
    if change > 5:
        return "strong_uptrend"
    elif change > 0:
        return "uptrend"
    elif change < -5:
        return "strong_downtrend"
    else:
        return "downtrend"

def _generate_signal(coin_data: Dict) -> str:
    """تولید سیگنال"""
    change = coin_data.get('priceChange1d', 0)
    volume = coin_data.get('volume', 0)
    
    if change > 3 and volume > 1000000:
        return "BUY"
    elif change < -3 and volume > 1000000:
        return "SELL"
    else:
        return "HOLD"

def _calculate_confidence(coin_data: Dict) -> float:
    """محاسبه اعتماد"""
    volume = coin_data.get('volume', 0)
    market_cap = coin_data.get('marketCap', 0)
    
    base_confidence = 0.5
    volume_boost = min(0.3, volume / 10000000000)
    market_cap_boost = min(0.2, market_cap / 1000000000000)
    
    return round(base_confidence + volume_boost + market_cap_boost, 2)

def _analyze_chart_trend(prices: List) -> str:
    """تحلیل روند چارت"""
    if len(prices) < 2:
        return "unknown"
    
    first_price = prices[0][1] if len(prices[0]) > 1 else prices[0]
    last_price = prices[-1][1] if len(prices[-1]) > 1 else prices[-1]
    
    if last_price > first_price:
        return "uptrend"
    else:
        return "downtrend"

def _calculate_volatility(prices: List) -> float:
    """محاسبه نوسان"""
    if len(prices) < 2:
        return 0.0
    
    price_values = [p[1] if len(p) > 1 else p for p in prices]
    avg_price = sum(price_values) / len(price_values)
    variance = sum((p - avg_price) ** 2 for p in price_values) / len(price_values)
    
    return round((variance ** 0.5) / avg_price * 100, 2)

def _find_support_resistance(prices: List) -> Dict:
    """پیدا کردن سطوح حمایت و مقاومت"""
    if len(prices) < 10:
        return {'support': 0, 'resistance': 0}
    
    price_values = [p[1] if len(p) > 1 else p for p in prices]
    
    return {
        'support': min(price_values),
        'resistance': max(price_values),
        'current_range': f"{min(price_values)} - {max(price_values)}"
    }
