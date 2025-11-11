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

insights_router = APIRouter(prefix="/api/insights", tags=["Insights"])

@insights_router.get("/btc-dominance", summary="دامیننس بیت‌کوین")
@insights_cache
async def get_btc_dominance(type: str = Query("all")):
    """دریافت دامیننس بیت‌کوین پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_btc_dominance(type)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        processed_data = {
            'period': type,
            'dominance_percentage': raw_data.get('dominance'),
            'trend': _analyze_dominance_trend(raw_data),
            'market_implication': _get_market_implication(raw_data),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in BTC dominance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@insights_router.get("/fear-greed", summary="شاخص ترس و طمع")
@insights_cache
async def get_fear_greed():
    """دریافت شاخص ترس و طمع پردازش شده"""
    try:
        # دریافت داده خام مستقیم از API
        raw_data = coin_stats_manager._make_api_request("insights/fear-and-greed")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار جدید API
        if "now" in raw_data:
            current_data = raw_data["now"]
            value = current_data.get('value')
            value_classification = current_data.get('value_classification')
            timestamp = current_data.get('timestamp')
        else:
            # Fallback برای ساختار قدیمی
            value = raw_data.get('value')
            value_classification = raw_data.get('value_classification') 
            timestamp = raw_data.get('timestamp')
        
        # اگر داده معتبر نیست، از مقادیر پیش‌فرض استفاده کن
        if value is None:
            value = 50
            value_classification = "Neutral"
            timestamp = datetime.now().timestamp()
        
        # تحلیل داده
        analysis = _analyze_fear_greed_value(value)
        recommendation = _get_fear_greed_recommendation(value)
        
        processed_data = {
            'value': value,
            'value_classification': value_classification,
            'timestamp': timestamp,
            'time_until_update': None,
            'analysis': analysis,
            'recommendation': recommendation,
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in fear-greed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@insights_router.get("/fear-greed/chart", summary="چارت ترس و طمع")
@insights_cache
async def get_fear_greed_chart():
    """دریافت چارت ترس و طمع پردازش شده"""
    try:
        # استفاده از متد manager
        raw_data = coin_stats_manager.get_fear_greed_chart()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌های چارت
        chart_data = raw_data.get('data', []) if isinstance(raw_data, dict) else raw_data
        
        processed_chart = {
            'data': chart_data,
            'analysis': _analyze_fear_greed_trend(chart_data),
            'period': 'historical',
            'total_data_points': len(chart_data),
            'date_range': _get_date_range(chart_data),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_chart,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in fear-greed chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@insights_router.get("/rainbow-chart/{coin_id}", summary="چارت رنگین‌کمان")
@insights_cache
async def get_rainbow_chart(coin_id: str):
    """دریافت چارت رنگین‌کمان پردازش شده"""
    try:
        # استفاده از متد manager
        raw_data = coin_stats_manager.get_rainbow_chart(coin_id)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌های rainbow chart
        chart_data = raw_data if isinstance(raw_data, list) else raw_data.get('data', [])
        
        processed_chart = {
            'coin_id': coin_id,
            'data': chart_data,
            'analysis': _analyze_rainbow_chart(chart_data),
            'signal': _generate_rainbow_signal(chart_data),
            'total_points': len(chart_data),
            'price_range': _get_price_range(chart_data),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_chart,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in rainbow chart for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================== توابع کمکی پردازش ==========================

def _process_fear_greed_data(raw_data: Dict) -> Dict:
    """پردازش داده‌های Fear & Greed"""
    if "now" in raw_data:
        current_data = raw_data["now"]
        value = current_data.get('value', 50)
    else:
        value = raw_data.get('value', 50)
    
    return {
        'value': value,
        'value_classification': _get_classification(value),
        'timestamp': datetime.now().isoformat(),
        'analysis': _analyze_fear_greed_value(value),
        'recommendation': _get_fear_greed_recommendation(value),
        'market_sentiment': _get_market_sentiment(value)
    }

def _get_classification(value: int) -> str:
    """دریافت طبقه‌بندی بر اساس مقدار"""
    if value >= 75:
        return "Extreme Greed"
    elif value >= 55:
        return "Greed"
    elif value >= 45:
        return "Neutral"
    elif value >= 25:
        return "Fear"
    else:
        return "Extreme Fear"

def _analyze_fear_greed_value(value: int) -> Dict[str, str]:
    """تحلیل مقدار Fear & Greed"""
    if value >= 75:
        return {"sentiment": "extreme_greed", "risk_level": "high", "market_condition": "Overbought"}
    elif value >= 55:
        return {"sentiment": "greed", "risk_level": "medium", "market_condition": "Optimistic"}
    elif value >= 45:
        return {"sentiment": "neutral", "risk_level": "medium", "market_condition": "Balanced"}
    elif value >= 25:
        return {"sentiment": "fear", "risk_level": "medium", "market_condition": "Cautious"}
    else:
        return {"sentiment": "extreme_fear", "risk_level": "high", "market_condition": "Oversold"}

def _get_fear_greed_recommendation(value: int) -> str:
    """تولید توصیه بر اساس مقدار"""
    if value >= 75:
        return "CAUTION: Market is overbought, consider taking profits"
    elif value >= 55:
        return "OPTIMISTIC: Good for holding, watch for opportunities"
    elif value >= 45:
        return "NEUTRAL: Good for accumulation and long-term holding"
    elif value >= 25:
        return "CAUTIOUS: Look for buying opportunities in quality assets"
    else:
        return "OPPORTUNITY: Market is oversold, potential for rebounds"

def _get_market_sentiment(value: int) -> str:
    """دریافت احساسات بازار"""
    if value >= 60:
        return "bullish"
    elif value >= 40:
        return "neutral"
    else:
        return "bearish"

def _analyze_fear_greed_trend(chart_data: List) -> Dict[str, Any]:
    """تحلیل روند شاخص ترس و طمع"""
    if len(chart_data) < 2:
        return {'trend': 'insufficient_data', 'momentum': 'neutral', 'direction': 'unknown'}
    
    recent_values = [point.get('value', 50) for point in chart_data[-10:]]  # ۱۰ نقطه آخر
    if len(recent_values) < 2:
        return {'trend': 'insufficient_data', 'momentum': 'neutral'}
    
    current_value = recent_values[-1]
    previous_value = recent_values[0]
    
    # تحلیل روند
    if current_value > previous_value + 5:
        trend = "improving"
        momentum = "bullish"
    elif current_value < previous_value - 5:
        trend = "deteriorating"
        momentum = "bearish"
    else:
        trend = "stable"
        momentum = "neutral"
    
    # تحلیل جهت کلی
    if current_value >= 55:
        direction = "positive"
    elif current_value <= 45:
        direction = "negative"
    else:
        direction = "neutral"
    
    return {
        'trend': trend,
        'momentum': momentum,
        'direction': direction,
        'current_value': current_value,
        'average_recent': round(sum(recent_values) / len(recent_values), 1),
        'volatility': round(max(recent_values) - min(recent_values), 1)
    }

def _analyze_rainbow_chart(chart_data: List) -> Dict[str, Any]:
    """تحلیل چارت رنگین‌کمان"""
    if not chart_data:
        return {'signal': 'no_data', 'zone': 'unknown', 'trend': 'unknown'}
    
    latest_point = chart_data[-1] if chart_data else {}
    price = float(latest_point.get('price', 0))
    
    # تحلیل بر اساس سطوح قیمتی (مقادیر نمونه - نیاز به تنظیم دقیق‌تر)
    if price > 120000:
        zone = "Bubble Territory"
        signal = "EXTREME_SELL"
        risk = "very_high"
    elif price > 80000:
        zone = "Sell Zone"
        signal = "SELL"
        risk = "high"
    elif price > 50000:
        zone = "Hold Zone"
        signal = "HOLD"
        risk = "medium"
    elif price > 30000:
        zone = "Buy Zone"
        signal = "BUY"
        risk = "low"
    else:
        zone = "Strong Buy Zone"
        signal = "STRONG_BUY"
        risk = "very_low"
    
    # تحلیل روند
    if len(chart_data) >= 5:
        recent_prices = [float(point.get('price', 0)) for point in chart_data[-5:]]
        price_change = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100
        if price_change > 5:
            trend = "bullish"
        elif price_change < -5:
            trend = "bearish"
        else:
            trend = "neutral"
    else:
        trend = "unknown"
    
    return {
        'current_zone': zone,
        'signal': signal,
        'risk_level': risk,
        'trend': trend,
        'price_level': price,
        'analysis_date': latest_point.get('time', 'unknown')
    }

def _generate_rainbow_signal(chart_data: List) -> str:
    """تولید سیگنال از چارت رنگین‌کمان"""
    analysis = _analyze_rainbow_chart(chart_data)
    return analysis.get('signal', 'HOLD')

def _get_date_range(chart_data: List) -> Dict[str, str]:
    """دریافت محدوده تاریخ داده‌ها"""
    if not chart_data:
        return {'start': 'unknown', 'end': 'unknown'}
    
    dates = [point.get('timestamp') for point in chart_data if point.get('timestamp')]
    if dates:
        return {'start': min(dates), 'end': max(dates)}
    return {'start': 'unknown', 'end': 'unknown'}

def _get_price_range(chart_data: List) -> Dict[str, float]:
    """دریافت محدوده قیمت"""
    if not chart_data:
        return {'min': 0, 'max': 0, 'current': 0}
    
    prices = [float(point.get('price', 0)) for point in chart_data]
    current_price = prices[-1] if prices else 0
    return {
        'min': min(prices) if prices else 0,
        'max': max(prices) if prices else 0,
        'current': current_price
    }

def _analyze_dominance_trend(dominance_data: Dict) -> str:
    """تحلیل روند دامیننس"""
    dominance = dominance_data.get('dominance', 50)
    
    if dominance > 55:
        return "bitcoin_dominant"
    elif dominance > 45:
        return "balanced"
    else:
        return "altcoin_season"

def _get_market_implication(dominance_data: Dict) -> str:
    """دریافت implications بازار"""
    dominance = dominance_data.get('dominance', 50)
    
    if dominance > 60:
        return "Bitcoin is dominating, altcoins may underperform"
    elif dominance < 40:
        return "Altcoin season likely, Bitcoin may underperform"
    else:
        return "Balanced market, watch for sector rotation"
