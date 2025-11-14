from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

# 🔧 اصلاح ایمپورت
try:
    from debug_system.storage.cache_decorators import cache_raw_insights_with_archive
    logger.info("✅ Cache System: Archive Enabled")
except ImportError as e:
    logger.error(f"❌ Cache system unavailable: {e}")
    # Fallback نهایی
    def cache_raw_insights_with_archive():
        def decorator(func):
            return func
        return decorator


raw_insights_router = APIRouter(prefix="/api/raw/insights", tags=["Raw Insights"])

@raw_insights_router.get("/btc-dominance", summary="داده‌های دامیننس بیت‌کوین")
@cache_raw_insights_with_archive()
async def get_raw_btc_dominance(type: str = Query("all", description="بازه زمانی: all, 24h, 1w, 1m, 3m, 1y")):
    """دریافت داده‌های خام دامیننس بیت‌کوین از CoinStats API"""
    try:
        # اگر type خالی است، مقدار پیش‌فرض تنظیم کن
        if not type or type.strip() == "":
            type = "all"
            
        raw_data = coin_stats_manager.get_btc_dominance(type)
        
        if not raw_data or "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data.get("error", "No data available"))
        
        # تحلیل داده‌های دامیننس
        dominance_analysis = _analyze_btc_dominance_data(raw_data, type)
        
        return {
            'status': 'success',
            'data_type': 'raw_btc_dominance',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'period_type': type,
            'timestamp': datetime.now().isoformat(),
            'analysis': dominance_analysis,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw BTC dominance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@raw_insights_router.get("/fear-greed", summary="داده‌های شاخص ترس و طمع")
@cache_raw_insights_with_archive()
async def get_raw_fear_greed():
    """دریافت داده‌های خام شاخص ترس و طمع از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_fear_greed()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # تحلیل شاخص ترس و طمع
        fear_greed_analysis = _analyze_fear_greed_data(raw_data)
        
        return {
            'status': 'success',
            'data_type': 'raw_fear_greed_index',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'analysis': fear_greed_analysis,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw fear-greed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_insights_router.get("/fear-greed/chart", summary="داده‌های تاریخی شاخص ترس و طمع")
@cache_raw_insights_with_archive()
async def get_raw_fear_greed_chart():
    """دریافت داده‌های تاریخی خام شاخص ترس و طمع از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_fear_greed_chart()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # تحلیل داده‌های تاریخی
        historical_analysis = _analyze_fear_greed_historical(raw_data)
        
        return {
            'status': 'success',
            'data_type': 'raw_fear_greed_historical',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'analysis': historical_analysis,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw fear-greed chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@raw_insights_router.get("/rainbow-chart/{coin_id}", summary="داده‌های چارت رنگین‌کمان")
@cache_raw_insights_with_archive()
async def get_raw_rainbow_chart(coin_id: str):
    """دریافت داده‌های خام چارت رنگین‌کمان از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_rainbow_chart(coin_id)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # تحلیل چارت رنگین‌کمان (ساده‌شده)
        rainbow_analysis = {
            'coin_id': coin_id,
            'data_points_count': len(raw_data.get('data', [])),
            'analysis_timestamp': datetime.now().isoformat(),
            'note': 'تحلیل پیشرفته در نسخه‌های آینده اضافه خواهد شد'
        }
        
        return {
            'status': 'success',
            'data_type': 'raw_rainbow_chart',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'coin_id': coin_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': rainbow_analysis,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw rainbow chart for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
def _analyze_rainbow_chart_data(rainbow_data: Dict, coin_id: str) -> Dict[str, Any]:
    """تحلیل داده‌های چارت رنگین‌کمان"""
    data_points = rainbow_data.get('data', [])
    
    if not data_points:
        return {'analysis': 'no_rainbow_chart_data_available'}
    
    # استخراج داده‌های قیمتی با تبدیل به float
    prices = []
    for point in data_points:
        if isinstance(point, dict):
            price = point.get('price')
            if price is not None:
                try:
                    # تبدیل قیمت به عدد
                    if isinstance(price, str):
                        price = float(price.replace(',', ''))
                    else:
                        price = float(price)
                    prices.append(price)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert price to float: {price}, error: {e}")
                    continue
    
    if not prices:
        return {
            'analysis': 'no_valid_price_data_in_rainbow_chart',
            'coin_id': coin_id,
            'data_points_received': len(data_points),
            'data_sample': data_points[:3] if data_points else []
        }
    
    current_price = prices[-1] if prices else 0
    min_price = min(prices)
    max_price = max(prices)
    
    # تحلیل موقعیت در چرخه بازار
    cycle_position = _analyze_market_cycle_position(current_price, min_price, max_price)
    
    return {
        'coin_id': coin_id,
        'data_points_count': len(data_points),
        'valid_price_points': len(prices),
        'price_analysis': {
            'current_price': current_price,
            'historical_min': min_price,
            'historical_max': max_price,
            'price_range_percentage': ((current_price - min_price) / (max_price - min_price)) * 100 if max_price > min_price else 0
        },
        'market_cycle_analysis': cycle_position,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
@raw_insights_router.get("/market-analysis", summary="تحلیل کلی بازار")
@cache_raw_insights_with_archive()
async def get_market_analysis():
    """دریافت تحلیل جامع بازار از داده‌های واقعی - برای آموزش هوش مصنوعی"""
    try:
        # جمع‌آوری داده‌های مختلف برای تحلیل جامع
        btc_dominance_data = coin_stats_manager.get_btc_dominance("all")
        fear_greed_data = coin_stats_manager.get_fear_greed()
        
        market_analysis = _perform_comprehensive_market_analysis(
            btc_dominance_data, 
            fear_greed_data
        )
        
        return {
            'status': 'success',
            'data_type': 'comprehensive_market_analysis',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'market_analysis': market_analysis,
            'data_sources': {
                'btc_dominance': btc_dominance_data,
                'fear_greed_index': fear_greed_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_insights_router.get("/metadata", summary="متادیتای بینش‌های بازار")
@cache_raw_insights_with_archive()
async def get_insights_metadata():
    """دریافت متادیتای کامل بینش‌های بازار - برای آموزش هوش مصنوعی"""
    try:
        return {
            'status': 'success',
            'data_type': 'insights_metadata',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'available_endpoints': [
                {
                    'endpoint': '/api/raw/insights/btc-dominance',
                    'description': 'داده‌های دامیننس بیت‌کوین در بازه‌های زمانی مختلف',
                    'parameters': ['type (all, 24h, 1w, 1m, 3m, 1y)'],
                    'use_case': 'تحلیل سلطه بازار و شناسایی فصل آلت‌کوین‌ها'
                },
                {
                    'endpoint': '/api/raw/insights/fear-greed',
                    'description': 'شاخص فعلی ترس و طمع بازار',
                    'parameters': [],
                    'use_case': 'تحلیل احساسات بازار و شناسایی نقاط چرخش'
                },
                {
                    'endpoint': '/api/raw/insights/fear-greed/chart',
                    'description': 'داده‌های تاریخی شاخص ترس و طمع',
                    'parameters': [],
                    'use_case': 'تحلیل روند احساسات بازار در طول زمان'
                },
                {
                    'endpoint': '/api/raw/insights/rainbow-chart/{coin_id}',
                    'description': 'داده‌های چارت رنگین‌کمان برای تحلیل قیمت',
                    'parameters': ['coin_id'],
                    'use_case': 'تحلیل سطوح قیمتی و شناسایی مناطق خرید/فروش'
                },
                {
                    'endpoint': '/api/raw/insights/market-analysis',
                    'description': 'تحلیل جامع بازار از چندین منبع',
                    'parameters': [],
                    'use_case': 'تحلیل چند بعدی بازار برای تصمیم‌گیری‌های پیچیده'
                }
            ],
            'analytical_metrics': {
                'btc_dominance': {
                    'description': 'درصد سلطه بیت‌کوین در کل بازار ارزهای دیجیتال',
                    'interpretation': 'مقادیر بالا نشان‌دهنده سلطه بیت‌کوین، مقادیر پایین نشان‌دهنده فصل آلت‌کوین‌ها',
                    'typical_range': '40% - 70%'
                },
                'fear_greed_index': {
                    'description': 'شاخص احساسات بازار از 0 (ترس شدید) تا 100 (طمع شدید)',
                    'interpretation': 'مقادیر پایین نشان‌دهنده فرصت خرید، مقادیر بالا نشان‌دهنده خطر اصلاح',
                    'zones': {
                        '0-24': 'ترس شدید',
                        '25-44': 'ترس',
                        '45-55': 'خنثی',
                        '56-75': 'طمع',
                        '76-100': 'طمع شدید'
                    }
                },
                'rainbow_chart': {
                    'description': 'تحلیل تکنیکال پیشرفته برای شناسایی چرخه‌های بازار',
                    'use_cases': ['شناسایی مناطق اشباع خرید/فروش', 'تحلیل چرخه‌های بلندمدت']
                }
            },
            'field_descriptions': _get_insights_field_descriptions()
        }
        
    except Exception as e:
        logger.error(f"Error in insights metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================ توابع کمکی برای هوش مصنوعی ============================

def _analyze_btc_dominance_data(dominance_data: Dict, period_type: str) -> Dict[str, Any]:
    """تحلیل داده‌های دامیننس بیت‌کوین"""
    dominance_value = dominance_data.get('dominance')
    
    if dominance_value is None:
        return {'analysis': 'no_dominance_data_available'}
    
    # تحلیل بر اساس مقادیر استاندارد بازار
    market_phase = "unknown"
    if dominance_value > 55:
        market_phase = "bitcoin_dominance"
        implication = "بیت‌کوین در حال سلطه بر بازار است - آلت‌کوین‌ها ممکن است عملکرد ضعیفی داشته باشند"
    elif dominance_value > 45:
        market_phase = "balanced_market"
        implication = "بازار متعادل - نوبت به نوبت بیت‌کوین و آلت‌کوین‌ها"
    else:
        market_phase = "altcoin_season"
        implication = "فصل آلت‌کوین‌ها - آلت‌کوین‌ها ممکن است outperform کنند"
    
    return {
        'current_dominance': dominance_value,
        'market_phase': market_phase,
        'market_implication': implication,
        'period_analyzed': period_type,
        'analysis_timestamp': datetime.now().isoformat(),
        'trading_suggestion': _get_dominance_trading_suggestion(dominance_value)
    }

def _analyze_fear_greed_data(fear_greed_data: Dict) -> Dict[str, Any]:
    """تحلیل داده‌های شاخص ترس و طمع"""
    value = fear_greed_data.get('value')
    classification = fear_greed_data.get('value_classification', '')
    
    if value is None:
        return {'analysis': 'no_fear_greed_data_available'}
    
    # تحلیل احساسات بازار
    sentiment_analysis = _classify_market_sentiment(value)
    
    return {
        'current_index': value,
        'classification': classification,
        'sentiment_analysis': sentiment_analysis,
        'market_condition': _get_market_condition(value),
        'risk_level': _get_risk_level(value),
        'historical_context': _get_historical_context(value),
        'analysis_timestamp': datetime.now().isoformat()
    }

def _analyze_fear_greed_historical(historical_data: Dict) -> Dict[str, Any]:
    """تحلیل داده‌های تاریخی شاخص ترس و طمع"""
    data_points = historical_data.get('data', [])
    
    if not data_points:
        return {'analysis': 'no_historical_data_available'}
    
    # استخراج مقادیر تاریخی
    values = [point.get('value', 0) for point in data_points if point.get('value') is not None]
    timestamps = [point.get('timestamp') for point in data_points]
    
    if not values:
        return {'analysis': 'insufficient_historical_data'}
    
    # محاسبات آماری
    current_value = values[-1] if values else 0
    average_value = sum(values) / len(values)
    min_value = min(values)
    max_value = max(values)
    
    # تحلیل روند
    if len(values) >= 10:
        recent_avg = sum(values[-10:]) / 10
        previous_avg = sum(values[-20:-10]) / 10 if len(values) >= 20 else average_value
        trend = "improving" if recent_avg > previous_avg else "deteriorating" if recent_avg < previous_avg else "stable"
    else:
        trend = "insufficient_data"
    
    return {
        'data_points_count': len(data_points),
        'time_period_covered': f"{len(data_points)} points",
        'current_value': current_value,
        'statistical_analysis': {
            'average': round(average_value, 2),
            'minimum': min_value,
            'maximum': max_value,
            'volatility': round(max_value - min_value, 2)
        },
        'trend_analysis': {
            'direction': trend,
            'current_sentiment': _classify_market_sentiment(current_value),
            'extreme_events': len([v for v in values if v <= 25 or v >= 75])
        },
        'analysis_timestamp': datetime.now().isoformat()
    }

def _analyze_rainbow_chart_data(rainbow_data: Dict, coin_id: str) -> Dict[str, Any]:
    """تحلیل داده‌های چارت رنگین‌کمان"""
    data_points = rainbow_data.get('data', [])
    
    if not data_points:
        return {'analysis': 'no_rainbow_chart_data_available'}
    
    # استخراج داده‌های قیمتی
    prices = []
    for point in data_points:
        if isinstance(point, dict):
            price = point.get('price')
            if price is not None:
                prices.append(price)
    
    if not prices:
        return {'analysis': 'no_price_data_in_rainbow_chart'}
    
    current_price = prices[-1] if prices else 0
    min_price = min(prices)
    max_price = max(prices)
    
    # تحلیل موقعیت در چرخه بازار
    cycle_position = _analyze_market_cycle_position(current_price, min_price, max_price)
    
    return {
        'coin_id': coin_id,
        'data_points_count': len(data_points),
        'price_analysis': {
            'current_price': current_price,
            'historical_min': min_price,
            'historical_max': max_price,
            'price_range_percentage': ((current_price - min_price) / (max_price - min_price)) * 100 if max_price > min_price else 0
        },
        'market_cycle_analysis': cycle_position,
        'analysis_timestamp': datetime.now().isoformat()
    }

def _perform_comprehensive_market_analysis(btc_dominance: Dict, fear_greed: Dict) -> Dict[str, Any]:
    """انجام تحلیل جامع بازار از چندین منبع"""
    dominance_value = btc_dominance.get('dominance')
    fear_greed_value = fear_greed.get('value')
    
    analysis = {
        'market_health_score': 0,
        'primary_trend': 'unknown',
        'risk_assessment': 'unknown',
        'trading_environment': 'unknown',
        'key_insights': []
    }
    
    # تحلیل بر اساس دامیننس
    if dominance_value is not None:
        if dominance_value > 55:
            analysis['primary_trend'] = 'bitcoin_led'
            analysis['key_insights'].append('بازار تحت سلطه بیت‌کوین است')
        elif dominance_value < 45:
            analysis['primary_trend'] = 'altcoin_season'
            analysis['key_insights'].append('شرایط مناسب برای آلت‌کوین‌ها')
        else:
            analysis['primary_trend'] = 'balanced'
            analysis['key_insights'].append('بازار در حالت تعادل')
    
    # تحلیل بر اساس شاخص ترس و طمع
    if fear_greed_value is not None:
        if fear_greed_value <= 25:
            analysis['risk_assessment'] = 'low_risk_high_opportunity'
            analysis['trading_environment'] = 'accumulation_phase'
            analysis['key_insights'].append('احساسات بازار به منطقه ترس رسیده - فرصت خرید')
        elif fear_greed_value >= 75:
            analysis['risk_assessment'] = 'high_risk_caution'
            analysis['trading_environment'] = 'distribution_phase'
            analysis['key_insights'].append('احساسات بازار به منطقه طمع رسیده - احتیاط لازم')
        else:
            analysis['risk_assessment'] = 'moderate_risk'
            analysis['trading_environment'] = 'normal_trading'
            analysis['key_insights'].append('احساسات بازار در محدوده طبیعی')
    
    # محاسبه امتیاز سلامت بازار
    health_score = 50  # پایه
    
    if dominance_value is not None and fear_greed_value is not None:
        # منطق ساده برای امتیازدهی
        if 40 <= dominance_value <= 60:
            health_score += 20
        if 30 <= fear_greed_value <= 70:
            health_score += 30
    
    analysis['market_health_score'] = min(health_score, 100)
    
    return analysis

def _classify_market_sentiment(value: float) -> Dict[str, Any]:
    """طبقه‌بندی احساسات بازار"""
    if value <= 24:
        return {
            'zone': 'extreme_fear',
            'sentiment': 'very_bearish',
            'color': 'red',
            'description': 'ترس شدید - بازار ممکن است oversold باشد'
        }
    elif value <= 44:
        return {
            'zone': 'fear',
            'sentiment': 'bearish', 
            'color': 'orange',
            'description': 'ترس - احساسات منفی غالب است'
        }
    elif value <= 55:
        return {
            'zone': 'neutral',
            'sentiment': 'neutral',
            'color': 'yellow',
            'description': 'خنثی - بازار در تعادل'
        }
    elif value <= 75:
        return {
            'zone': 'greed',
            'sentiment': 'bullish',
            'color': 'light_green',
            'description': 'طمع - احساسات مثبت در حال رشد'
        }
    else:
        return {
            'zone': 'extreme_greed',
            'sentiment': 'very_bullish',
            'color': 'green',
            'description': 'طمع شدید - بازار ممکن است overbought باشد'
        }

def _get_market_condition(value: float) -> str:
    """دریافت شرایط بازار"""
    if value <= 25:
        return "بازار نزولی - فرصت‌های خرید بالقوه"
    elif value <= 45:
        return "بازار محتاط - انتظار برای جهت‌گیری"
    elif value <= 55:
        return "بازار متعادل - معاملات عادی"
    elif value <= 75:
        return "بازار صعودی - روند مثبت"
    else:
        return "بازار گرم - خطر اصلاح قیمت"

def _get_risk_level(value: float) -> str:
    """دریافت سطح ریسک"""
    if value <= 25 or value >= 75:
        return "high"
    elif value <= 35 or value >= 65:
        return "medium_high"
    elif value <= 45 or value >= 55:
        return "medium"
    else:
        return "low"

def _get_historical_context(value: float) -> str:
    """دریافت زمینه تاریخی"""
    if value <= 20:
        return "سطح بسیار پایین - مشابه کف‌های تاریخی بازار"
    elif value >= 80:
        return "سطح بسیار بالا - مشابه سقف‌های تاریخی بازار"
    else:
        return "سطح نرمال - در محدوده متعارف تاریخی"

def _get_dominance_trading_suggestion(dominance: float) -> str:
    """پیشنهاد معاملاتی بر اساس دامیننس"""
    if dominance > 60:
        return "تمرکز بر بیت‌کوین - آلت‌کوین‌ها ممکن است تحت فشار باشند"
    elif dominance < 40:
        return "فرصت‌های آلت‌کوین - فصل آلت‌کوین‌ها ممکن است فعال باشد"
    else:
        return "تعادل بازار - تنوع‌بخشی مناسب است"

def _analyze_market_cycle_position(current_price: float, min_price: float, max_price: float) -> Dict[str, Any]:
    """تحلیل موقعیت در چرخه بازار"""
    try:
        if max_price <= min_price:
            return {
                'position': 'unknown', 
                'phase': 'insufficient_data',
                'error': 'max_price <= min_price'
            }
        
        position_percentage = ((current_price - min_price) / (max_price - min_price)) * 100
        
        if position_percentage <= 20:
            phase = "accumulation"
            suggestion = "منطقه خرید - قیمت در کف تاریخی"
        elif position_percentage <= 40:
            phase = "early_uptrend" 
            suggestion = "روند صعودی اولیه - فرصت‌های خوب"
        elif position_percentage <= 60:
            phase = "mid_cycle"
            suggestion = "میانه چرخه - روند ثابت"
        elif position_percentage <= 80:
            phase = "late_uptrend"
            suggestion = "انتهای روند صعودی - احتیاط"
        else:
            phase = "distribution"
            suggestion = "منطقه فروش - قیمت در سقف تاریخی"
        
        return {
            'position_percentage': round(position_percentage, 2),
            'market_phase': phase,
            'trading_suggestion': suggestion,
            'risk_level': 'high' if position_percentage >= 80 else 'low' if position_percentage <= 20 else 'medium'
        }
    except Exception as e:
        logger.error(f"Error in market cycle analysis: {e}")
        return {
            'position': 'error',
            'phase': 'calculation_error',
            'error_message': str(e)
        }

def _get_insights_field_descriptions() -> Dict[str, str]:
    """توضیحات فیلدهای بینش‌های بازار"""
    return {
        'dominance': 'درصد سلطه بیت‌کوین در کل بازار ارزهای دیجیتال',
        'value': 'مقدار شاخص ترس و طمع (0-100)',
        'value_classification': 'طبقه‌بندی احساسات بازار',
        'timestamp': 'زمان ثبت داده',
        'time_until_update': 'زمان تا بروزرسانی بعدی',
        'data': 'داده‌های تاریخی در قالب آرایه',
        'price': 'قیمت در چارت رنگین‌کمان'
    }
