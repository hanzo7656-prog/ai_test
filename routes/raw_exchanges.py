from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

raw_exchanges_router = APIRouter(prefix="/api/raw/exchanges", tags=["Raw Exchanges"])

@raw_exchanges_router.get("/list", summary="لیست خام صرافی‌ها")
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
        else:
            currencies_list = raw_data.get('data', raw_data.get('result', []))
        
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
        logger.error(f"Error in raw currencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_exchanges_router.get("/metadata", summary="متادیتای صرافی‌ها و مارکت‌ها")
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
    """تحلیل داده‌های واقعی صرافی‌ها"""
    if not exchanges:
        return {}
    
    # استخراج داده‌های عددی
    trust_scores = [ex.get('trust_score', 0) for ex in exchanges if ex.get('trust_score')]
    volumes_24h = [ex.get('trade_volume_24h_btc', 0) for ex in exchanges if ex.get('trade_volume_24h_btc')]
    years_established = [ex.get('year_established', 0) for ex in exchanges if ex.get('year_established')]
    
    # تحلیل کشورها
    countries = {}
    for ex in exchanges:
        country = ex.get('country', 'Unknown')
        countries[country] = countries.get(country, 0) + 1
    
    return {
        'total_exchanges': len(exchanges),
        'trust_score_stats': {
            'min': min(trust_scores) if trust_scores else 0,
            'max': max(trust_scores) if trust_scores else 0,
            'average': sum(trust_scores) / len(trust_scores) if trust_scores else 0,
            'high_trust': len([s for s in trust_scores if s >= 8])
        },
        'volume_stats': {
            'total_volume_24h_btc': sum(volumes_24h) if volumes_24h else 0,
            'average_volume': sum(volumes_24h) / len(volumes_24h) if volumes_24h else 0,
            'top_exchanges': len([v for v in volumes_24h if v > 1000])  # صرافی‌های با حجم بالا
        },
        'establishment_stats': {
            'oldest': min(years_established) if years_established else 0,
            'newest': max(years_established) if years_established else 0,
            'average_year': sum(years_established) / len(years_established) if years_established else 0
        },
        'geographical_distribution': {
            'total_countries': len(countries),
            'top_countries': dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5])
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

def _analyze_fiats_data(fiats: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی ارزهای فیات"""
    if not fiats:
        return {}
    
    symbols = [f.get('symbol', '') for f in fiats]
    decimal_digits = [f.get('decimal_digits', 0) for f in fiats if f.get('decimal_digits') is not None]
    
    return {
        'total_fiats': len(fiats),
        'symbol_stats': {
            'unique_symbols': len(set(symbols)),
            'common_symbols': [sym for sym in set(symbols) if symbols.count(sym) > 1]
        },
        'formatting_stats': {
            'average_decimals': sum(decimal_digits) / len(decimal_digits) if decimal_digits else 0,
            'min_decimals': min(decimal_digits) if decimal_digits else 0,
            'max_decimals': max(decimal_digits) if decimal_digits else 0
        }
    }

def _analyze_currencies_data(currencies: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی ارزها"""
    if not currencies:
        return {}
    
    # این endpoint ممکن است ساختار متفاوتی داشته باشد
    return {
        'total_currencies': len(currencies),
        'data_structure_sample': currencies[0] if currencies else {},
        'available_fields': list(currencies[0].keys()) if currencies else []
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
    """توضیحات فیلدهای صرافی‌ها و مارکت‌ها"""
    return {
        # فیلدهای صرافی
        'id': 'شناسه یکتا صرافی',
        'name': 'نام صرافی',
        'year_established': 'سال تأسیس',
        'country': 'کشور محل استقرار',
        'trust_score': 'امتیاز اعتماد (1-10)',
        'trade_volume_24h_btc': 'حجم معاملات 24 ساعته به بیت‌کوین',
        'url': 'آدرس وبسایت',
        'image': 'آدرس لوگو',
        
        # فیلدهای مارکت
        'exchangeId': 'شناسه صرافی',
        'baseAsset': 'ارز پایه',
        'quoteAsset': 'ارز متقابل',
        'price': 'قیمت فعلی',
        'volume24h': 'حجم معاملات 24 ساعته',
        
        # فیلدهای فیات
        'symbol': 'نماد ارز',
        'name': 'نام ارز',
        'symbol_native': 'نماد محلی',
        'decimal_digits': 'تعداد اعشار',
        'rounding': 'گرد کردن',
        'code': 'کد ارز',
        'name_plural': 'نام جمع'
    }

def _get_exchange_field_description(field: str) -> str:
    """توضیحات فیلدهای صرافی"""
    descriptions = {
        'id': 'شناسه یکتای صرافی در سیستم',
        'name': 'نام رسمی صرافی',
        'year_established': 'سال تأسیس صرافی',
        'country': 'کشور محل ثبت صرافی',
        'trust_score': 'امتیاز اعتماد از 1 تا 10',
        'trade_volume_24h_btc': 'حجم معاملات 24 ساعته به بیت‌کوین',
        'url': 'آدرس وبسایت رسمی',
        'image': 'آدرس تصویر لوگو'
    }
    return descriptions.get(field, 'فیلد اطلاعات صرافی')

def _get_market_field_description(field: str) -> str:
    """توضیحات فیلدهای مارکت"""
    descriptions = {
        'exchangeId': 'شناسه صرافی میزبان مارکت',
        'baseAsset': 'ارز پایه در جفت معاملاتی',
        'quoteAsset': 'ارز متقابل در جفت معاملاتی',
        'price': 'قیمت فعلی جفت ارز',
        'volume24h': 'حجم معاملات 24 ساعته'
    }
    return descriptions.get(field, 'فیلد اطلاعات مارکت')
