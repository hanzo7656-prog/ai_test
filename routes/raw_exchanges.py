from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

# 🔧 اصلاح ایمپورت
try:
    from debug_system.storage.cache_decorators import cache_raw_exchanges_with_archive
    logger.info("✅ Cache System: Archive Enabled")
except ImportError as e:
    logger.error(f"❌ Cache system unavailable: {e}")
    # Fallback نهایی
    def cache_raw_exchanges_with_archive():
        def decorator(func):
            return func
        return decorator


raw_exchanges_router = APIRouter(prefix="/api/raw/exchanges", tags=["Raw Exchanges"])

@raw_exchanges_router.get("/list", summary="لیست خام صرافی‌ها")
@cache_raw_exchanges_with_archive()
async def get_raw_exchanges_list():
    """دریافت لیست خام صرافی‌ها از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # دریافت مستقیم از API - دور زدن متد مشکل‌دار
        raw_data = coin_stats_manager._make_api_request("tickers/exchanges")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            exchanges_list = raw_data
        else:
            exchanges_list = raw_data.get('data', raw_data.get('result', []))
        
        # تحلیل داده‌های واقعی صرافی‌ها
        exchange_stats = _analyze_exchanges_data(exchanges_list)
        
        return {
            'status': 'success',
            'data_type': 'raw_exchanges_list',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'statistics': exchange_stats,
            'total_exchanges': len(exchanges_list),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw exchanges list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/markets", summary="داده‌های مارکت‌ها")
@cache_raw_exchanges_with_archive()
async def get_raw_markets():
    """دریافت داده‌های خام مارکت‌ها از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # دریافت مستقیم از API
        raw_data = coin_stats_manager._make_api_request("tickers/markets")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            markets_list = raw_data
        else:
            markets_list = raw_data.get('data', raw_data.get('result', []))
        
        market_stats = _analyze_markets_data(markets_list)
        
        return {
            'status': 'success',
            'data_type': 'raw_markets_data',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'statistics': market_stats,
            'total_markets': len(markets_list),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/fiats", summary="داده‌های ارزهای فیات")
@cache_raw_exchanges_with_archive()
async def get_raw_fiats():
    """دریافت داده‌های خام ارزهای فیات از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # دریافت مستقیم از API - دور زدن متد مشکل‌دار
        raw_data = coin_stats_manager._make_api_request("fiats")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            fiats_list = raw_data
        else:
            fiats_list = raw_data.get('data', raw_data.get('result', []))
        
        fiat_stats = _analyze_fiats_data(fiats_list)
        
        return {
            'status': 'success',
            'data_type': 'raw_fiats_data',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'statistics': fiat_stats,
            'total_fiats': len(fiats_list),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw fiats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/currencies", summary="داده‌های ارزها")
@cache_raw_exchanges_with_archive()
async def get_raw_currencies():
    """دریافت داده‌های خام ارزها از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # دریافت مستقیم از API
        raw_data = coin_stats_manager._make_api_request("currencies")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            currencies_list = raw_data
        elif isinstance(raw_data, dict):
            # ساختارهای مختلف API
            if 'data' in raw_data:
                currencies_list = raw_data['data']
            elif 'result' in raw_data:
                currencies_list = raw_data['result']
            else:
                # اگر داده مستقیماً در ریشه است
                currencies_list = [raw_data]
        else:
            currencies_list = []
        
        # اگر currencies_list هنوز None یا خالی است
        if not currencies_list:
            currencies_list = []
        
        currency_stats = _analyze_currencies_data(currencies_list)
        
        return {
            'status': 'success',
            'data_type': 'raw_currencies_data',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'statistics': currency_stats,
            'total_currencies': len(currencies_list),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw currencies: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Error processing currencies data: {str(e)}")
@raw_exchanges_router.get("/metadata", summary="متادیتای صرافی‌ها و مارکت‌ها")
@cache_raw_exchanges_with_archive()
async def get_exchanges_metadata():
    """دریافت متادیتای کامل صرافی‌ها و مارکت‌ها - برای آموزش هوش مصنوعی"""
    try:
        # دریافت مستقیم از API برای داده نمونه
        sample_exchanges = coin_stats_manager._make_api_request("tickers/exchanges")
        sample_markets = coin_stats_manager._make_api_request("tickers/markets")
        
        exchanges_structure = {}
        markets_structure = {}
        
        if not "error" in sample_exchanges:
            exchanges_list = sample_exchanges if isinstance(sample_exchanges, list) else sample_exchanges.get('data', [])
            if exchanges_list:
                exchanges_structure = _extract_exchange_structure(exchanges_list[0])
        
        if not "error" in sample_markets:
            markets_list = sample_markets if isinstance(sample_markets, list) else sample_markets.get('data', [])
            if markets_list:
                markets_structure = _extract_market_structure(markets_list[0])
        
        return {
            'status': 'success',
            'data_type': 'exchanges_metadata',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'available_endpoints': [
                {
                    'endpoint': '/api/raw/exchanges/list',
                    'description': 'لیست کامل صرافی‌ها',
                    'data_type': 'exchanges_list'
                },
                {
                    'endpoint': '/api/raw/exchanges/markets',
                    'description': 'داده‌های مارکت‌ها و جفت‌ارزها',
                    'data_type': 'markets_data'
                },
                {
                    'endpoint': '/api/raw/exchanges/fiats',
                    'description': 'داده‌های ارزهای فیات',
                    'data_type': 'fiats_data'
                },
                {
                    'endpoint': '/api/raw/exchanges/currencies',
                    'description': 'داده‌های ارزهای دیجیتال',
                    'data_type': 'currencies_data'
                }
            ],
            'data_structures': {
                'exchange_structure': exchanges_structure,
                'market_structure': markets_structure
            },
            'field_descriptions': _get_exchanges_field_descriptions()
        }
        
    except Exception as e:
        logger.error(f"Error in exchanges metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/exchange/{exchange_id}", summary="داده‌های اختصاصی صرافی")
@cache_raw_exchanges_with_archive()
async def get_exchange_details(exchange_id: str):
    """دریافت داده‌های اختصاصی یک صرافی - برای تحلیل‌های پیشرفته هوش مصنوعی"""
    try:
        # دریافت تمام صرافی‌ها و فیلتر بر اساس ID
        raw_data = coin_stats_manager.get_exchanges()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        exchanges_list = raw_data if isinstance(raw_data, list) else raw_data.get('data', [])
        
        # جستجوی صرافی مورد نظر
        target_exchange = None
        for exchange in exchanges_list:
            if exchange.get('id') == exchange_id or exchange.get('name', '').lower() == exchange_id.lower():
                target_exchange = exchange
                break
        
        if not target_exchange:
            raise HTTPException(status_code=404, detail=f"صرافی با شناسه {exchange_id} یافت نشد")
        
        # تحلیل داده‌های صرافی
        exchange_analysis = _analyze_single_exchange(target_exchange)
        
        return {
            'status': 'success',
            'data_type': 'exchange_details',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'exchange_id': exchange_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': exchange_analysis,
            'data': target_exchange
        }
        
    except Exception as e:
        logger.error(f"Error in exchange details for {exchange_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================ توابع کمکی برای هوش مصنوعی ============================

def _analyze_exchanges_data(exchanges: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی صرافی‌ها با فیلدهای موجود"""
    if not exchanges:
        return {}
    
    # استفاده از فیلدهای واقعی موجود
    volumes_24h = [ex.get('volume24h', 0) for ex in exchanges if ex.get('volume24h')]
    volumes_7d = [ex.get('volume7d', 0) for ex in exchanges if ex.get('volume7d')]
    changes_24h = [ex.get('change24h', 0) for ex in exchanges if ex.get('change24h')]
    ranks = [ex.get('rank', 0) for ex in exchanges if ex.get('rank')]
    
    return {
        'total_exchanges': len(exchanges),
        'volume_stats_24h': {
            'total': sum(volumes_24h) if volumes_24h else 0,
            'average': sum(volumes_24h) / len(volumes_24h) if volumes_24h else 0,
            'min': min(volumes_24h) if volumes_24h else 0,
            'max': max(volumes_24h) if volumes_24h else 0
        },
        'performance_stats': {
            'positive_24h': len([c for c in changes_24h if c > 0]),
            'negative_24h': len([c for c in changes_24h if c < 0]),
            'average_change': sum(changes_24h) / len(changes_24h) if changes_24h else 0
        },
        'rank_distribution': {
            'top_10': len([r for r in ranks if r <= 10]),
            'top_50': len([r for r in ranks if r <= 50]),
            'average_rank': sum(ranks) / len(ranks) if ranks else 0
        }
    }
def _analyze_markets_data(markets: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی مارکت‌ها"""
    if not markets:
        return {}
    
    # استخراج جفت‌ارزها و قیمت‌ها
    base_assets = {}
    quote_assets = {}
    prices = [m.get('price', 0) for m in markets if m.get('price')]
    volumes = [m.get('volume24h', 0) for m in markets if m.get('volume24h')]
    
    for market in markets:
        base = market.get('baseAsset')
        quote = market.get('quoteAsset')
        
        if base:
            base_assets[base] = base_assets.get(base, 0) + 1
        if quote:
            quote_assets[quote] = quote_assets.get(quote, 0) + 1
    
    return {
        'total_markets': len(markets),
        'price_stats': {
            'min': min(prices) if prices else 0,
            'max': max(prices) if prices else 0,
            'average': sum(prices) / len(prices) if prices else 0
        },
        'volume_stats': {
            'total_volume_24h': sum(volumes) if volumes else 0,
            'average_volume': sum(volumes) / len(volumes) if volumes else 0
        },
        'asset_distribution': {
            'unique_base_assets': len(base_assets),
            'unique_quote_assets': len(quote_assets),
            'top_base_assets': dict(sorted(base_assets.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_quote_assets': dict(sorted(quote_assets.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    }

def _analyze_markets_data(markets: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی مارکت‌ها با فیلدهای موجود"""
    if not markets:
        return {}
    
    # استفاده از فیلدهای واقعی
    base_assets = {}
    quote_assets = {}
    prices = [m.get('price', 0) for m in markets if m.get('price')]
    volumes = [m.get('volume', 0) for m in markets if m.get('volume')]
    
    for market in markets:
        base = market.get('from')  # ✅ استفاده از فیلد واقعی
        quote = market.get('to')   # ✅ استفاده از فیلد واقعی
        
        if base:
            base_assets[base] = base_assets.get(base, 0) + 1
        if quote:
            quote_assets[quote] = quote_assets.get(quote, 0) + 1
    
    return {
        'total_markets': len(markets),
        'price_stats': {
            'min': min(prices) if prices else 0,
            'max': max(prices) if prices else 0,
            'average': sum(prices) / len(prices) if prices else 0
        },
        'volume_stats': {
            'total_volume': sum(volumes) if volumes else 0,
            'average_volume': sum(volumes) / len(volumes) if volumes else 0
        },
        'asset_distribution': {
            'unique_base_assets': len(base_assets),
            'unique_quote_assets': len(quote_assets),
            'top_base_assets': dict(sorted(base_assets.items(), key=lambda x: x[1], reverse=True)[:5]),
            'top_quote_assets': dict(sorted(quote_assets.items(), key=lambda x: x[1], reverse=True)[:3])
        }
    }

def _analyze_currencies_data(currencies: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی ارزها"""
    if not currencies:
        return {'total_currencies': 0, 'analysis': 'no_data'}
    
    try:
        # تحلیل ساختار داده‌ها
        if isinstance(currencies, dict):
            # اگر currencies یک دیکشنری است (مانند نرخ تبدیل)
            currency_codes = list(currencies.keys())
            rates = list(currencies.values())
            
            return {
                'total_currencies': len(currency_codes),
                'data_type': 'exchange_rates',
                'rate_stats': {
                    'min_rate': min(rates) if rates else 0,
                    'max_rate': max(rates) if rates else 0,
                    'average_rate': sum(rates) / len(rates) if rates else 0,
                    'base_currency': 'USD'  # فرض می‌کنیم پایه USD است
                },
                'available_currencies': currency_codes[:10]  # 10 ارز اول
            }
        
        elif isinstance(currencies, list):
            # اگر currencies یک لیست است
            if not currencies:
                return {'total_currencies': 0, 'analysis': 'empty_list'}
            
            # بررسی ساختار اولین آیتم
            first_item = currencies[0] if currencies else {}
            
            if isinstance(first_item, dict):
                # تحلیل فیلدهای موجود
                available_fields = list(first_item.keys()) if first_item else []
                
                # شمارش انواع داده
                field_types = {}
                for field in available_fields:
                    sample_value = first_item.get(field)
                    field_types[field] = type(sample_value).__name__
                
                return {
                    'total_currencies': len(currencies),
                    'data_structure': 'list_of_objects',
                    'available_fields': available_fields,
                    'field_types': field_types,
                    'sample_data': first_item
                }
            else:
                # اگر لیست از مقادیر ساده است
                return {
                    'total_currencies': len(currencies),
                    'data_structure': 'simple_list',
                    'sample_values': currencies[:5]  # 5 مقدار اول
                }
        
        else:
            return {
                'total_currencies': 0,
                'data_structure': 'unknown',
                'raw_data_type': type(currencies).__name__
            }
            
    except Exception as e:
        logger.error(f"Error in currencies analysis: {str(e)}")
        return {
            'total_currencies': len(currencies) if currencies else 0,
            'analysis_error': str(e),
            'data_type': 'error_in_analysis'
        }
def _analyze_single_exchange(exchange: Dict) -> Dict[str, Any]:
    """تحلیل داده‌های یک صرافی خاص"""
    analysis = {
        'reliability_score': exchange.get('trust_score', 0),
        'trading_volume': exchange.get('trade_volume_24h_btc', 0),
        'establishment_year': exchange.get('year_established'),
        'country': exchange.get('country', 'Unknown'),
        'has_image': bool(exchange.get('image')),
        'has_url': bool(exchange.get('url'))
    }
    
    # رتبه‌بندی بر اساس امتیاز اعتماد
    trust_score = exchange.get('trust_score', 0)
    if trust_score >= 9:
        analysis['reliability_rating'] = 'excellent'
    elif trust_score >= 7:
        analysis['reliability_rating'] = 'good'
    elif trust_score >= 5:
        analysis['reliability_rating'] = 'moderate'
    else:
        analysis['reliability_rating'] = 'low'
    
    # تحلیل حجم معاملات
    volume = exchange.get('trade_volume_24h_btc', 0)
    if volume > 10000:
        analysis['volume_rating'] = 'very_high'
    elif volume > 1000:
        analysis['volume_rating'] = 'high'
    elif volume > 100:
        analysis['volume_rating'] = 'medium'
    else:
        analysis['volume_rating'] = 'low'
    
    return analysis

def _extract_exchange_structure(exchange: Dict) -> Dict[str, Any]:
    """استخراج ساختار داده صرافی"""
    structure = {}
    
    for key, value in exchange.items():
        structure[key] = {
            'type': type(value).__name__,
            'sample_value': value if not isinstance(value, (dict, list)) or not value else 'complex_data',
            'description': _get_exchange_field_description(key)
        }
    
    return structure

def _extract_market_structure(market: Dict) -> Dict[str, Any]:
    """استخراج ساختار داده مارکت"""
    structure = {}
    
    for key, value in market.items():
        structure[key] = {
            'type': type(value).__name__,
            'sample_value': value if not isinstance(value, (dict, list)) or not value else 'complex_data',
            'description': _get_market_field_description(key)
        }
    
    return structure

def _get_exchanges_field_descriptions() -> Dict[str, str]:
    """توضیحات فیلدهای واقعی صرافی‌ها و مارکت‌ها"""
    return {
        # ==================== فیلدهای صرافی‌ها ====================
        'id': 'شناسه یکتا صرافی در سیستم CoinStats',
        'name': 'نام رسمی صرافی',
        'rank': 'رتبه صرافی بر اساس حجم معاملات (هرچه کمتر، بهتر)',
        'volume24h': 'حجم معاملات 24 ساعته (USD)',
        'volume7d': 'حجم معاملات 7 روزه (USD)',
        'volume1m': 'حجم معاملات 1 ماهه (USD)',
        'change24h': 'تغییر حجم معاملات 24 ساعته (درصد)',
        'url': 'آدرس وبسایت رسمی صرافی',
        'icon': 'آدرس تصویر لوگو صرافی',
        
        # ==================== فیلدهای مارکت‌ها ====================
        'exchange': 'نام صرافی میزبان مارکت',
        'from': 'ارز پایه در جفت معاملاتی (ارز فروخته شده)',
        'to': 'ارز متقابل در جفت معاملاتی (ارز خریداری شده)',
        'pair': 'نماد جفت معاملاتی (فرمت: BASE/QUOTE)',
        'price': 'قیمت فعلی جفت ارز',
        'pairPrice': 'قیمت جفت ارز (ممکن است با price متفاوت باشد)',
        'volume': 'حجم معاملات 24 ساعته (USD)',
        'pairVolume': 'حجم معاملات بر اساس ارز پایه',
        '_created_at': 'تاریخ ایجاد رکورد در دیتابیس (ISO format)',
        '_updated_at': 'تاریخ آخرین بروزرسانی داده (ISO format)',
        
        # ==================== فیلدهای ارزهای فیات ====================
        'symbol': 'نماد ارز فیات (مانند $ برای USD)',
        'symbol_native': 'نماد محلی ارز فیات',
        'decimal_digits': 'تعداد اعشار مجاز برای این ارز',
        'rounding': 'الگوی گرد کردن اعداد برای این ارز',
        'code': 'کد استاندارد ارز (ISO 4217)',
        'name_plural': 'نام جمع ارز (مانند dollars برای USD)',
        
        # ==================== فیلدهای ارزهای دیجیتال ====================
        'currency_code': 'کد ارز دیجیتال',
        'currency_name': 'نام کامل ارز دیجیتال',
        'rate': 'نرخ تبدیل به USD',
        'is_fiat': 'آیا ارز فیات است یا دیجیتال',
        'is_active': 'آیا ارز فعال است',
        
        # ==================== فیلدهای عمومی ====================
        'timestamp': 'زمان تولید پاسخ (ISO format)',
        'status': 'وضعیت درخواست (success/error)',
        'data_type': 'نوع داده بازگشتی',
        'source': 'منبع داده (coinstats_api)',
        'api_version': 'نسخه API استفاده شده',
        
        # ==================== فیلدهای آماری ====================
        'total_exchanges': 'تعداد کل صرافی‌های بازگشتی',
        'total_markets': 'تعداد کل مارکت‌های بازگشتی',
        'total_fiats': 'تعداد کل ارزهای فیات',
        'total_currencies': 'تعداد کل ارزهای دیجیتال',
        
        # ==================== فیلدهای تحلیل ====================
        'price_stats': 'آمار قیمت‌ها (min, max, average)',
        'volume_stats': 'آمار حجم معاملات',
        'asset_distribution': 'توزیع ارزهای پایه و متقابل',
        'performance_stats': 'آمار عملکرد صرافی‌ها',
        'rank_distribution': 'توزیع رتبه‌های صرافی‌ها',
        
        # ==================== فیلدهای متادیتا ====================
        'data_structure': 'ساختار داده‌های بازگشتی',
        'available_endpoints': 'لیست endpointهای در دسترس',
        'field_descriptions': 'توضیحات فیلدهای موجود'
    }

def _get_exchange_field_description(field: str) -> str:
    """توضیحات فیلدهای صرافی"""
    descriptions = {
        'id': 'شناسه یکتای صرافی در سیستم CoinStats - برای استفاده در API calls',
        'name': 'نام رسمی و شناخته شده صرافی در بازار',
        'rank': 'رتبه صرافی بر اساس حجم معاملات 24 ساعته - عدد کمتر نشان‌دهنده رتبه بهتر است',
        'volume24h': 'حجم کل معاملات 24 ساعته صرافی به دلار آمریکا (USD)',
        'volume7d': 'حجم کل معاملات 7 روز گذشته صرافی به دلار آمریکا',
        'volume1m': 'حجم کل معاملات 30 روز گذشته صرافی به دلار آمریکا',
        'change24h': 'درصد تغییر حجم معاملات نسبت به 24 ساعت قبل - مثبت نشان‌دهنده رشد است',
        'url': 'آدرس کامل وبسایت رسمی صرافی برای دسترسی مستقیم',
        'icon': 'آدرس تصویر لوگو صرافی با کیفیت مناسب برای نمایش',
        
        # فیلدهای اضافی که ممکن است در آینده اضافه شوند
        'trust_score': 'امتیاز اعتماد صرافی از 1 تا 10 (در صورت موجود بودن)',
        'year_established': 'سال تأسیس صرافی (در صورت موجود بودن)',
        'country': 'کشور محل ثبت و فعالیت صرافی (در صورت موجود بودن)',
        'trading_pairs': 'تعداد جفت‌ارزهای فعال در صرافی (در صورت موجود بودن)',
        'has_trading_incentive': 'آیا صرافی incentive معاملاتی ارائه می‌دهد',
        'centralized': 'آیا صرافی متمرکز است (true/false)',
        'public_notice': 'اطلاعیه عمومی مربوط به صرافی',
        'alert_notice': 'هشدارهای امنیتی مربوط به صرافی'
    }
    return descriptions.get(field, 'فیلد اطلاعات صرافی')

def _get_market_field_description(field: str) -> str:
    """توضیحات فیلدهای مارکت"""
    descriptions = {
        'exchange': 'نام صرافی میزبان این مارکت معاملاتی',
        'from': 'ارز پایه (base currency) - ارزی که خرید و فروش می‌شود',
        'to': 'ارز متقابل (quote currency) - ارزی که برای قیمت‌گذاری استفاده می‌شود',
        'pair': 'نماد کامل جفت معاملاتی به فرمت استاندارد BASE/QUOTE',
        'price': 'آخرین قیمت معامله شده این جفت ارز',
        'pairPrice': 'قیمت جفت ارز - ممکن است در برخی موارد با price متفاوت باشد',
        'volume': 'حجم کل معاملات 24 ساعته این جفت ارز به دلار آمریکا',
        'pairVolume': 'حجم معاملات بر اساس واحد ارز پایه',
        '_created_at': 'تاریخ و زمان اولین ذخیره‌سازی این رکورد در دیتابیس',
        '_updated_at': 'تاریخ و زمان آخرین بروزرسانی این داده',
        
        # فیلدهای اضافی
        'last_updated': 'زمان آخرین بروزرسانی قیمت',
        'price_change_24h': 'تغییر قیمت 24 ساعته (درصد)',
        'price_change_7d': 'تغییر قیمت 7 روزه (درصد)',
        'market_cap': 'ارزش بازار این جفت ارز (در صورت محاسبه)',
        'liquidity': 'نقدینگی بازار (در صورت موجود بودن)',
        'spread': 'اختلاف قیمت خرید و فروش (bid-ask spread)'
    }
    return descriptions.get(field, 'فیلد اطلاعات مارکت')

def _get_fiat_field_description(field: str) -> str:
    """توضیحات فیلدهای ارزهای فیات"""
    descriptions = {
        'symbol': 'نماد گرافیکی ارز (مانند $، €، £)',
        'symbol_native': 'نماد محلی ارز در کشور مبدأ',
        'decimal_digits': 'تعداد رقم‌های اعشار مجاز برای این ارز',
        'rounding': 'الگوریتم گرد کردن اعداد برای این ارز',
        'code': 'کد سه حرفی استاندارد ISO 4217 (مانند USD, EUR, GBP)',
        'name_plural': 'نام جمع ارز برای استفاده در متون',
        
        # فیلدهای اضافی
        'name': 'نام رسمی و کامل ارز',
        'symbol_position': 'موقعیت نماد نسبت به عدد (before/after)',
        'space_between': 'آیا بین نماد و عدد فاصله وجود دارد',
        'decimal_separator': 'جداکننده اعشار (معمولاً نقطه یا کاما)',
        'thousands_separator': 'جداکننده هزارگان',
        'smallest_denomination': 'کوچکترین واحد پول قابل معامله'
    }
    return descriptions.get(field, 'فیلد اطلاعات ارز فیات')
