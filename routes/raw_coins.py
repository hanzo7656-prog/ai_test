from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

try:
    from debug_system.storage.smart_cache_system import coins_cache
    logger.info("✅ Using Smart Cache for coins")
except ImportError as e:
    logger.warning(f"⚠️ Smart Cache not available: {e}")
    try:
        # fallback به سیستم قدیم
        from debug_system.storage.cache_decorators import cache_coins as coins_cache
        logger.info("✅ Using Legacy Cache for coins")
    except ImportError as e2:
        logger.error(f"❌ No cache system available: {e2}")
        # تعریف دکوراتور خالی به عنوان fallback نهایی
        def coins_cache(func):
            return func

raw_coins_router = APIRouter(prefix="/api/raw/coins", tags=["Raw Coins"])

@raw_coins_router.get("/list", summary="لیست خام نمادها")
@raw_coins_cache
async def get_raw_coins_list(
    limit: int = Query(20, ge=1, le=1000),
    page: int = Query(1, ge=1),
    currency: str = Query("USD"),
    sort_by: str = Query("rank"),
    sort_dir: str = Query("asc"),
    coin_ids: str = Query(None, description="فیلتر بر اساس coin_idها (bitcoin,ethereum,...)"),
    name: str = Query(None, description="جستجو در نام کوین‌ها"),
    symbol: str = Query(None, description="فیلتر بر اساس نماد (BTC,ETH,...)"),
    blockchains: str = Query(None, description="فیلتر بر اساس بلاکچین‌ها (ethereum,solana,...)"),
    categories: str = Query(None, description="فیلتر بر اساس دسته‌بندی‌ها (defi,gaming,...)")
):
    """دریافت لیست خام نمادها از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # ساخت پارامترهای فیلتر
        filters = {}
        if coin_ids:
            filters['coinIds'] = coin_ids
        if name:
            filters['name'] = name
        if symbol:
            filters['symbol'] = symbol
        if blockchains:
            filters['blockchains'] = blockchains
        if categories:
            filters['categories'] = categories
        
        raw_data = coin_stats_manager.get_coins_list(
            limit=limit, 
            page=page, 
            currency=currency, 
            sort_by=sort_by,
            sort_dir=sort_dir,
            **filters
        )
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # محاسبه آمار واقعی از داده‌ها
        coins = raw_data.get('data', [])
        stats = _calculate_real_stats(coins)
        
        return {
            'status': 'success',
            'data_type': 'raw_coins_list',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'pagination': raw_data.get('meta', {}),
            'filters_applied': {
                'limit': limit,
                'page': page,
                'currency': currency,
                'sort_by': sort_by,
                'sort_dir': sort_dir,
                **filters
            },
            'statistics': stats,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw coins list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/details/{coin_id}", summary="جزئیات خام نماد")
@raw_coins_cache
async def get_raw_coin_details(
    coin_id: str, 
    currency: str = Query("USD"),
    include_risk_score: bool = Query(False, description="شامل داده‌های ریسک")
):
    """دریافت جزئیات کامل یک نماد از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # ساخت پارامترها
        params = {"currency": currency}
        if include_risk_score:
            params['includeRiskScore'] = "true"
        
        raw_data = coin_stats_manager.get_coin_details(coin_id, currency)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data_type': 'raw_coin_details',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'coin_id': coin_id,
            'currency': currency,
            'timestamp': datetime.now().isoformat(),
            'data_structure': _get_coin_data_structure(raw_data),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw coin details for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/charts/{coin_id}", summary="داده‌های چارت نماد")
@raw_coins_cache
async def get_raw_coin_charts(
    coin_id: str, 
    period: str = Query("all", description="بازه زمانی: 24h,1w,1m,3m,1y,all")
):
    """دریافت داده‌های تاریخی چارت از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # دریافت مستقیم از API - دور زدن متد مشکل‌دار
        params = {"period": period, "coinIds": coin_id}
        raw_data = coin_stats_manager._make_api_request("coins/charts", params)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            chart_data = raw_data
        else:
            chart_data = raw_data.get("data", raw_data.get("result", []))
        
        # تحلیل داده‌های واقعی چارت
        chart_analysis = _analyze_chart_data(chart_data)
        
        return {
            'status': 'success',
            'data_type': 'raw_chart_data',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'coin_id': coin_id,
            'period': period,
            'timestamp': datetime.now().isoformat(),
            'chart_analysis': chart_analysis,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw coin charts for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/multi-charts", summary="داده‌های چارت چندنماد")
@raw_coins_cache
async def get_raw_multi_charts(
    coin_ids: str = Query(..., description="لیست coin_idها با کاما جدا شده (bitcoin,ethereum,solana)"),
    period: str = Query("all", description="بازه زمانی: 24h,1w,1m,3m,1y,all")
):
    """دریافت داده‌های تاریخی چند نماد از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        # دریافت مستقیم از API - دور زدن متد مشکل‌دار
        params = {"coinIds": coin_ids, "period": period}
        raw_data = coin_stats_manager._make_api_request("coins/charts", params)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        coin_list = coin_ids.split(',')
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            chart_data = raw_data
        else:
            chart_data = raw_data.get("data", raw_data.get("result", []))
        
        return {
            'status': 'success',
            'data_type': 'raw_multi_chart_data',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'coin_ids': coin_list,
            'period': period,
            'coins_count': len(coin_list),
            'timestamp': datetime.now().isoformat(),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw multi-charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/price/avg", summary="قیمت متوسط تاریخی")
@raw_coins_cache
async def get_raw_coin_price_avg(
    coin_id: str = Query("bitcoin"),
    timestamp: str = Query("1636315200")
):
    """دریافت قیمت متوسط تاریخی از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_coin_price_avg(coin_id, timestamp)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data_type': 'raw_historical_price',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'coin_id': coin_id,
            'requested_timestamp': timestamp,
            'calculated_timestamp': datetime.now().isoformat(),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw price avg for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/price/exchange", summary="قیمت صرافی")
@raw_coins_cache
async def get_raw_exchange_price(
    exchange: str = Query("Binance"),
    from_coin: str = Query("BTC"),
    to_coin: str = Query("USDT"),
    timestamp: str = Query("1636315200")
):
    """دریافت قیمت مبادله از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_exchange_price(exchange, from_coin, to_coin, timestamp)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data_type': 'raw_exchange_price',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'exchange': exchange,
            'from_coin': from_coin,
            'to_coin': to_coin,
            'timestamp': timestamp,
            'calculated_timestamp': datetime.now().isoformat(),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw exchange price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_coins_router.get("/metadata", summary="متادیتای کوین‌ها")
@raw_coins_cache
async def get_coins_metadata():
    """دریافت متادیتای کامل کوین‌ها از CoinStats API - برای آموزش هوش مصنوعی"""
    try:
        # دریافت مستقیم از API برای داده نمونه
        raw_data = coin_stats_manager._make_api_request("coins", {"limit": 5})
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش ساختار واقعی API
        if isinstance(raw_data, list):
            coins_data = raw_data
        else:
            coins_data = raw_data.get("result", raw_data.get("data", []))
        
        if coins_data and len(coins_data) > 0:
            sample_coin = coins_data[0]
            data_structure = _extract_data_structure(sample_coin)
        else:
            data_structure = {"error": "No data available"}
        
        return {
            'status': 'success',
            'data_type': 'coins_metadata',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'available_endpoints': [
                {
                    'endpoint': '/api/raw/coins/list',
                    'description': 'لیست کامل کوین‌ها با فیلترهای پیشرفته',
                    'parameters': ['limit', 'page', 'currency', 'sort_by', 'sort_dir', 'coin_ids', 'name', 'symbol', 'blockchains', 'categories']
                },
                {
                    'endpoint': '/api/raw/coins/details/{coin_id}',
                    'description': 'جزئیات کامل یک کوین',
                    'parameters': ['coin_id', 'currency', 'include_risk_score']
                },
                {
                    'endpoint': '/api/raw/coins/charts/{coin_id}',
                    'description': 'داده‌های تاریخی چارت',
                    'parameters': ['coin_id', 'period']
                },
                {
                    'endpoint': '/api/raw/coins/multi-charts',
                    'description': 'داده‌های تاریخی چندین کوین',
                    'parameters': ['coin_ids', 'period']
                },
                {
                    'endpoint': '/api/raw/coins/price/avg',
                    'description': 'قیمت متوسط تاریخی',
                    'parameters': ['coin_id', 'timestamp']
                },
                {
                    'endpoint': '/api/raw/coins/price/exchange',
                    'description': 'قیمت مبادله در صرافی',
                    'parameters': ['exchange', 'from_coin', 'to_coin', 'timestamp']
                }
            ],
            'data_structure': data_structure,
            'field_descriptions': _get_field_descriptions()
        }
        
    except Exception as e:
        logger.error(f"Error in coins metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================ توابع کمکی برای هوش مصنوعی ============================

def _calculate_real_stats(coins: List[Dict]) -> Dict[str, Any]:
    """محاسبه آمار واقعی از داده‌های کوین‌ها"""
    if not coins:
        return {}
    
    prices = [coin.get('price', 0) for coin in coins if coin.get('price')]
    market_caps = [coin.get('marketCap', 0) for coin in coins if coin.get('marketCap')]
    volumes = [coin.get('volume', 0) for coin in coins if coin.get('volume')]
    changes_24h = [coin.get('priceChange1d', 0) for coin in coins if coin.get('priceChange1d')]
    
    return {
        'total_coins': len(coins),
        'price_stats': {
            'min': min(prices) if prices else 0,
            'max': max(prices) if prices else 0,
            'average': sum(prices) / len(prices) if prices else 0,
            'median': sorted(prices)[len(prices)//2] if prices else 0
        },
        'market_cap_stats': {
            'min': min(market_caps) if market_caps else 0,
            'max': max(market_caps) if market_caps else 0,
            'average': sum(market_caps) / len(market_caps) if market_caps else 0,
            'total': sum(market_caps) if market_caps else 0
        },
        'volume_stats': {
            'min': min(volumes) if volumes else 0,
            'max': max(volumes) if volumes else 0,
            'average': sum(volumes) / len(volumes) if volumes else 0,
            'total': sum(volumes) if volumes else 0
        },
        'performance_stats': {
            'positive_24h': len([c for c in changes_24h if c > 0]),
            'negative_24h': len([c for c in changes_24h if c < 0]),
            'neutral_24h': len([c for c in changes_24h if c == 0]),
            'average_change': sum(changes_24h) / len(changes_24h) if changes_24h else 0
        }
    }

def _analyze_chart_data(chart_data: List) -> Dict[str, Any]:
    """تحلیل داده‌های واقعی چارت"""
    if not chart_data or len(chart_data) < 2:
        return {'data_points': len(chart_data) if chart_data else 0, 'analysis': 'insufficient_data'}
    
    # استخراج مقادیر قیمت - فرض می‌کنیم ساختار [timestamp, price] دارد
    price_values = []
    timestamps = []
    
    for point in chart_data:
        if isinstance(point, list) and len(point) > 1:
            timestamps.append(point[0])
            price_values.append(point[1])
        elif isinstance(point, dict):
            # اگر ساختار دیکشنری است
            if 'price' in point:
                price_values.append(point['price'])
            if 'timestamp' in point:
                timestamps.append(point['timestamp'])
        elif isinstance(point, (int, float)):
            price_values.append(point)
    
    if len(price_values) < 2:
        return {'data_points': len(chart_data), 'analysis': 'invalid_price_data'}
    
    # محاسبات واقعی
    first_price = price_values[0]
    last_price = price_values[-1]
    min_price = min(price_values)
    max_price = max(price_values)
    total_change = ((last_price - first_price) / first_price) * 100 if first_price else 0
    
    volatility = 0
    if len(price_values) > 1:
        returns = []
        for i in range(1, len(price_values)):
            if price_values[i-1] != 0:
                returns.append((price_values[i] - price_values[i-1]) / price_values[i-1])
        
        if returns:
            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
    
    return {
        'data_points': len(chart_data),
        'price_points': len(price_values),
        'time_period': f"{len(price_values)} points",
        'price_range': {
            'first': first_price,
            'last': last_price,
            'min': min_price,
            'max': max_price,
            'total_change_percent': round(total_change, 2)
        },
        'volatility': round(volatility * 100, 4),  # نوسان به درصد
        'trend': 'up' if total_change > 0 else 'down' if total_change < 0 else 'flat'
    }

def _extract_data_structure(sample_coin: Dict) -> Dict[str, Any]:
    """استخراج ساختار داده از یک نمونه کوین"""
    structure = {}
    
    for key, value in sample_coin.items():
        if value is None:
            structure[key] = {'type': 'null', 'description': 'مقدار خالی'}
        elif isinstance(value, str):
            structure[key] = {'type': 'string', 'description': 'متن'}
        elif isinstance(value, (int, float)):
            structure[key] = {'type': 'number', 'description': 'عدد'}
        elif isinstance(value, bool):
            structure[key] = {'type': 'boolean', 'description': 'صحیح/غلط'}
        elif isinstance(value, list):
            structure[key] = {'type': 'array', 'description': 'لیست', 'sample_size': len(value)}
        elif isinstance(value, dict):
            structure[key] = {'type': 'object', 'description': 'شیء', 'keys': list(value.keys())}
        else:
            structure[key] = {'type': 'unknown', 'description': 'نوع ناشناخته'}
    
    return structure

def _get_coin_data_structure(coin_data: Dict) -> Dict[str, Any]:
    """دریافت ساختار داده‌های کوین"""
    fields = {}
    
    for key, value in coin_data.items():
        fields[key] = {
            'type': type(value).__name__,
            'sample_value': value if not isinstance(value, (dict, list)) or not value else 'complex_data',
            'nullable': value is None
        }
    
    return fields

def _get_field_descriptions() -> Dict[str, str]:
    """توضیحات فیلدهای مهم برای هوش مصنوعی"""
    return {
        'id': 'شناسه یکتا کوین',
        'name': 'نام کامل کوین',
        'symbol': 'نماد معاملاتی',
        'rank': 'رتبه بر اساس ارزش بازار',
        'price': 'قیمت فعلی به USD',
        'priceBtc': 'قیمت به بیت‌کوین',
        'volume': 'حجم معاملات 24 ساعته',
        'marketCap': 'ارزش بازار',
        'availableSupply': 'عرضه در گردش',
        'totalSupply': 'عرضه کل',
        'priceChange1h': 'تغییر قیمت 1 ساعته (%)',
        'priceChange1d': 'تغییر قیمت 24 ساعته (%)',
        'priceChange1w': 'تغییر قیمت 7 روزه (%)',
        'websiteUrl': 'آدرس وبسایت',
        'redditUrl': 'لینک انجمن رددیت',
        'twitterUrl': 'لینک توییتر',
        'contractAddress': 'آدرس قرارداد هوشمند',
        'decimals': 'تعداد اعشار',
        'explorers': 'اکسپلوررهای بلاکچین'
    }
