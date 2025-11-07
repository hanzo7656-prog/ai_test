from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

insights_router = APIRouter(prefix="/api/insights", tags=["Insights"])

@insights_router.get("/btc-dominance", summary="دامیننس بیت‌کوین")
async def get_btc_dominance(period_type: str = Query("all")):
    """دریافت دامیننس بیت‌کوین پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_btc_dominance(period_type)
        
        processed_data = {
            'period': period_type,
            'dominance_percentage': raw_data.get('dominance'),
            'trend': _analyze_dominance_trend(raw_data),
            'market_implication': _get_market_implication(raw_data),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in BTC dominance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@insights_router.get("/fear-greed", summary="شاخص ترس و طمع")
async def get_fear_greed():
    """دریافت شاخص ترس و طمع پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_fear_greed()
        
        processed_data = {
            'value': raw_data.get('value'),
            'value_classification': raw_data.get('value_classification'),
            'timestamp': raw_data.get('timestamp'),
            'time_until_update': raw_data.get('time_until_update'),
            'analysis': _analyze_fear_greed(raw_data),
            'recommendation': _get_fear_greed_recommendation(raw_data),
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
async def get_fear_greed_chart():
    """دریافت چارت ترس و طمع پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_fear_greed_chart()
        
        processed_chart = {
            'data': raw_data.get('data', []),
            'analysis': _analyze_fear_greed_trend(raw_data),
            'period': 'historical',
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
async def get_rainbow_chart(coin_id: str):
    """دریافت چارت رنگین‌کمان پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_rainbow_chart(coin_id)
        
        processed_chart = {
            'coin_id': coin_id,
            'data': raw_data.get('data', []),
            'analysis': _analyze_rainbow_chart(raw_data),
            'signal': _generate_rainbow_signal(raw_data),
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

# توابع کمکی پردازش بینش
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

def _analyze_fear_greed(fear_greed_data: Dict) -> Dict[str, Any]:
    """تحلیل شاخص ترس و طمع"""
    value = fear_greed_data.get('value', 50)
    classification = fear_greed_data.get('value_classification', '').lower()
    
    analysis = {
        'current_sentiment': classification,
        'market_condition': '',
        'risk_level': '',
        'suggested_action': ''
    }
    
    if value >= 75:
        analysis.update({
            'market_condition': 'Extreme Greed - Market may be overbought',
            'risk_level': 'High',
            'suggested_action': 'Consider taking profits'
        })
    elif value >= 55:
        analysis.update({
            'market_condition': 'Greed - Bullish sentiment',
            'risk_level': 'Medium',
            'suggested_action': 'Monitor for entry points'
        })
    elif value >= 45:
        analysis.update({
            'market_condition': 'Neutral - Balanced market',
            'risk_level': 'Low',
            'suggested_action': 'Good for accumulation'
        })
    elif value >= 25:
        analysis.update({
            'market_condition': 'Fear - Bearish sentiment', 
            'risk_level': 'Medium',
            'suggested_action': 'Look for buying opportunities'
        })
    else:
        analysis.update({
            'market_condition': 'Extreme Fear - Market may be oversold',
            'risk_level': 'High', 
            'suggested_action': 'Potential buying opportunity'
        })
    
    return analysis

def _get_fear_greed_recommendation(fear_greed_data: Dict) -> str:
    """دریافت توصیه بر اساس شاخص ترس و طمع"""
    value = fear_greed_data.get('value', 50)
    
    if value >= 75:
        return "CAUTION: Market shows extreme greed. Consider reducing exposure."
    elif value >= 55:
        return "OPTIMISTIC: Greed phase. Good for holding, be ready to take profits."
    elif value >= 45:
        return "NEUTRAL: Balanced market. Good for strategic accumulation."
    elif value >= 25:
        return "CAUTIOUS: Fear phase. Look for quality assets at discount."
    else:
        return "OPPORTUNITY: Extreme fear. Potential for strong rebounds."

def _analyze_fear_greed_trend(chart_data: Dict) -> Dict[str, Any]:
    """تحلیل روند شاخص ترس و طمع"""
    data_points = chart_data.get('data', [])
    
    if len(data_points) < 2:
        return {'trend': 'insufficient_data', 'momentum': 'neutral'}
    
    recent_values = [point.get('value', 50) for point in data_points[-10:]]  # ۱۰ نقطه آخر
    avg_recent = sum(recent_values) / len(recent_values)
    
    if len(data_points) >= 20:
        older_values = [point.get('value', 50) for point in data_points[-20:-10]]
        avg_older = sum(older_values) / len(older_values)
        
        if avg_recent > avg_older + 5:
            trend = "improving"
            momentum = "bullish"
        elif avg_recent < avg_older - 5:
            trend = "deteriorating" 
            momentum = "bearish"
        else:
            trend = "stable"
            momentum = "neutral"
    else:
        trend = "unknown"
        momentum = "neutral"
    
    return {
        'trend': trend,
        'momentum': momentum,
        'average_sentiment': round(avg_recent, 1)
    }

def _analyze_rainbow_chart(rainbow_data: Dict) -> Dict[str, Any]:
    """تحلیل چارت رنگین‌کمان"""
    data_points = rainbow_data.get('data', [])
    
    if not data_points:
        return {'signal': 'no_data', 'zone': 'unknown'}
    
    latest_point = data_points[-1] if data_points else {}
    price = latest_point.get('price', 0)
    
    # منطق ساده برای تحلیل چارت رنگین‌کمان
    if price > 100000:  # مثال
        zone = "Bubble Territory"
        signal = "EXTREME_SELL"
    elif price > 50000:
        zone = "Sell Zone" 
        signal = "SELL"
    elif price > 30000:
        zone = "Accumulation Zone"
        signal = "HOLD"
    elif price > 10000:
        zone = "Buy Zone"
        signal = "BUY"
    else:
        zone = "Fire Sale"
        signal = "STRONG_BUY"
    
    return {
        'current_zone': zone,
        'signal': signal,
        'price_level': price
    }

def _generate_rainbow_signal(rainbow_data: Dict) -> str:
    """تولید سیگنال از چارت رنگین‌کمان"""
    analysis = _analyze_rainbow_chart(rainbow_data)
    return analysis.get('signal', 'HOLD')
