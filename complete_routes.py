# complete_routes.py - نسخه کامل و هماهنگ با سیستم‌های دیگر
from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
from debug_manager import debug_endpoint, debug_manager

# ایمپورت مدیران
from complete_coinstats_manager import coin_stats_manager
from lbank_websocket import get_websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# مدل‌های درخواست
class AlertRequest(BaseModel):
    symbol: str
    condition: str
    target_price: float
    alert_type: str = "price"

# دیتابیس ساده
alerts_db = {}

# WebSocket Manager
lbank_ws = get_websocket_manager()

def set_websocket_manager(ws_manager):
    """تنظیم WebSocket manager"""
    global lbank_ws
    lbank_ws = ws_manager

# ==================== روت‌های اصلی ====================

@router.get("/")
@debug_endpoint
async def root():
    """صفحه اصلی"""
    websocket_status = "connected" if lbank_ws and lbank_ws.is_connected() else "disconnected"
    active_pairs = len(lbank_ws.get_realtime_data()) if lbank_ws else 0
    
    return {
        "message": "AI Trading Assistant API - Complete Version", 
        "version": "4.0.0",
        "status": "running",
        "websocket_status": websocket_status,
        "active_pairs": active_pairs,
        "endpoints": {
            "health": "/health",
            "coins": {
                "list": "/coins/list",
                "details": "/coins/{coin_id}",
                "charts": "/coins/{coin_id}/charts",
                "multi_charts": "/coins/charts/multi",
                "price_avg": "/coins/price/avg",
                "exchange_price": "/coins/price/exchange"
            },
            "market": {
                "overview": "/market/overview",
                "exchanges": "/market/exchanges",
                "markets": "/market/markets",
                "fiats": "/market/fiats",
                "currencies": "/market/currencies"
            },
            "news": {
                "sources": "/news/sources",
                "all": "/news",
                "handpicked": "/news/handpicked",
                "trending": "/news/trending",
                "latest": "/news/latest",
                "bullish": "/news/bullish", 
                "bearish": "/news/bearish",
                "detail": "/news/{news_id}"
            },
            "insights": {
                "btc_dominance": "/insights/btc-dominance",
                "fear_greed": "/insights/fear-greed",
                "fear_greed_chart": "/insights/fear-greed/chart",
                "rainbow_chart": "/insights/rainbow-chart/{coin_id}"
            },
            "websocket": {
                "status": "/websocket/status",
                "data": "/websocket/data/{symbol}",
                "active_pairs": "/websocket/pairs/active"
            },
            "system": {
                "debug": "/system/debug",
                "cache_info": "/system/cache/info",
                "health": "/system/health"
            },
            "alerts": {
                "create": "/alerts/create",
                "list": "/alerts/list",
                "delete": "/alerts/{alert_id}"
            }
        }
    }

@router.get("/health")
@debug_endpoint
async def health_check():
    """سلامت سرویس"""
    websocket_connected = lbank_ws and lbank_ws.is_connected()
    active_pairs = len(lbank_ws.get_realtime_data()) if lbank_ws else 0
    
    # تست اتصال به CoinStats API
    api_connected = False
    try:
        test_data = coin_stats_manager.get_coins_list(limit=1)
        api_connected = bool(test_data and test_data.get('result'))
    except Exception as e:
        logger.error(f"API Health check failed: {e}")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "coinstats_api": "connected" if api_connected else "disconnected",
            "websocket": "connected" if websocket_connected else "disconnected",
            "cache_system": "active"
        },
        "metrics": {
            "active_websocket_pairs": active_pairs,
            "cache_files": coin_stats_manager.get_cache_info().get('total_files', 0)
        }
    }

# ==================== روت‌های کوین‌ها ====================

@router.get("/coins/list")
@debug_endpoint
async def get_coins_list(
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    currency: str = Query("USD"),
    sort_by: str = Query("rank", regex="^(rank|marketCap|price|volume|name|symbol)$"),
    sort_dir: str = Query("asc", regex="^(asc|desc)$")
):
    """دریافت لیست کوین‌ها"""
    try:
        data = coin_stats_manager.get_coins_list(
            limit=limit, 
            page=page, 
            currency=currency,
            sort_by=sort_by,
            sort_dir=sort_dir
        )
        
        if not data:
            raise HTTPException(status_code=404, detail="No data received from API")
            
        return data
        
    except Exception as e:
        logger.error(f"Error in coins list: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching coins: {str(e)}")

@router.get("/coins/{coin_id}")
@debug_endpoint
async def get_coin_details(
    coin_id: str,
    currency: str = Query("USD")
):
    """دریافت جزئیات کوین خاص"""
    try:
        data = coin_stats_manager.get_coin_details(coin_id, currency)
        if not data:
            raise HTTPException(status_code=404, detail="Coin not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in coin details for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/{coin_id}/charts")
@debug_endpoint
async def get_coin_charts(
    coin_id: str,
    period: str = Query("1w", regex="^(24h|1w|1m|3m|6m|1y|all)$")
):
    """دریافت چارت کوین"""
    try:
        data = coin_stats_manager.get_coin_charts(coin_id, period)
        if not data:
            raise HTTPException(status_code=404, detail="Chart data not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in coin charts for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/charts/multi")
@debug_endpoint
async def get_multi_coin_charts(
    coin_ids: str = Query(..., alias="coinIds", description="لیست کوین‌ها (مثلاً bitcoin,ethereum,solana)"),
    period: str = Query("1w", regex="^(24h|1w|1m|3m|6m|1y|all)$")
):
    """دریافت چارت چندکوینه"""
    try:
        data = coin_stats_manager.get_coins_charts(coin_ids, period)
        if not data:
            raise HTTPException(status_code=404, detail="Chart data not found")
        return data
    except Exception as e:
        logger.error(f"Error in multi coin charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/price/avg")
@debug_endpoint
async def get_coin_price_avg(
    coin_id: str = Query(..., alias="coinId", description="شناسه کوین"),
    timestamp: str = Query(..., description="تایم‌استمپ")
):
    """دریافت قیمت متوسط"""
    try:
        data = coin_stats_manager.get_coin_price_avg(coin_id, timestamp)
        if not data:
            raise HTTPException(status_code=404, detail="Price data not found")
        return data
    except Exception as e:
        logger.error(f"Error in price avg for {coin_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/price/exchange")
@debug_endpoint
async def get_exchange_price(
    exchange: str = Query(..., description="نام صرافی"),
    from_coin: str = Query(..., alias="from", description="ارز مبدأ"),
    to_coin: str = Query(..., alias="to", description="ارز مقصد"),
    timestamp: str = Query(..., description="تایم‌استمپ")
):
    """دریافت قیمت مبادله"""
    try:
        data = coin_stats_manager.get_exchange_price(exchange, from_coin, to_coin, timestamp)
        if not data:
            raise HTTPException(status_code=404, detail="Exchange price not found")
        return data
    except Exception as e:
        logger.error(f"Error in exchange price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های بازار و صرافی ====================

@router.get("/market/overview")
@debug_endpoint
async def market_overview():
    """نمای کلی بازار با داده‌های ترکیبی"""
    try:
        # داده‌های CoinStats
        top_coins = coin_stats_manager.get_coins_list(limit=10)
        btc_dominance = coin_stats_manager.get_btc_dominance()
        fear_greed = coin_stats_manager.get_fear_greed()
        
        # داده‌های WebSocket
        websocket_data = {}
        if lbank_ws and lbank_ws.is_connected():
            websocket_data = lbank_ws.get_realtime_data()
        
        response = {}
        
        if top_coins and 'result' in top_coins:
            response["top_coins"] = top_coins['result'][:5]
            response["total_coins"] = len(top_coins['result'])
        
        if btc_dominance:
            response["btc_dominance"] = btc_dominance
        
        if fear_greed:
            response["fear_greed"] = fear_greed
            
        if websocket_data:
            response["websocket_prices"] = {
                symbol: data.get('price', 0) 
                for symbol, data in list(websocket_data.items())[:5]
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/exchanges")
@debug_endpoint
async def get_exchanges():
    """دریافت لیست صرافی‌ها"""
    try:
        data = coin_stats_manager.get_tickers_exchanges()
        if not data:
            raise HTTPException(status_code=404, detail="Exchanges data not found")
        return data
    except Exception as e:
        logger.error(f"Error in exchanges: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/markets")
@debug_endpoint
async def get_markets():
    """دریافت لیست مارکت‌ها"""
    try:
        data = coin_stats_manager.get_tickers_markets()
        if not data:
            data = coin_stats_manager.get_markets()
        if not data:
            raise HTTPException(status_code=404, detail="Markets data not found")
        return data
    except Exception as e:
        logger.error(f"Error in markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/fiats")
@debug_endpoint
async def get_fiats():
    """دریافت ارزهای فیات"""
    try:
        data = coin_stats_manager.get_fiats()
        if not data:
            raise HTTPException(status_code=404, detail="Fiats data not found")
        return data
    except Exception as e:
        logger.error(f"Error in fiats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/currencies")
@debug_endpoint
async def get_currencies():
    """دریافت لیست ارزها"""
    try:
        data = coin_stats_manager.get_currencies()
        if not data:
            raise HTTPException(status_code=404, detail="Currencies data not found")
        return data
    except Exception as e:
        logger.error(f"Error in currencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های اخبار ====================

@router.get("/news/sources")
@debug_endpoint
async def get_news_sources():
    """دریافت منابع خبری"""
    try:
        data = coin_stats_manager.get_news_sources()
        if not data:
            raise HTTPException(status_code=404, detail="News sources not found")
        return data
    except Exception as e:
        logger.error(f"Error in news sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news")
@debug_endpoint
async def get_news(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار عمومی"""
    try:
        data = coin_stats_manager.get_news(limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="News not found")
        return data
    except Exception as e:
        logger.error(f"Error in news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/handpicked")
@debug_endpoint
async def get_news_handpicked(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار گلچین شده"""
    try:
        data = coin_stats_manager.get_news_by_type("handpicked", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Handpicked news not found")
        return data
    except Exception as e:
        logger.error(f"Error in handpicked news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/trending")
@debug_endpoint
async def get_news_trending(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار ترندینگ"""
    try:
        data = coin_stats_manager.get_news_by_type("trending", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Trending news not found")
        return data
    except Exception as e:
        logger.error(f"Error in trending news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/latest")
@debug_endpoint
async def get_news_latest(limit: int = Query(50, ge=1, le=100)):
    """دریافت آخرین اخبار"""
    try:
        data = coin_stats_manager.get_news_by_type("latest", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Latest news not found")
        return data
    except Exception as e:
        logger.error(f"Error in latest news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/bullish")
@debug_endpoint
async def get_news_bullish(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار صعودی"""
    try:
        data = coin_stats_manager.get_news_by_type("bullish", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Bullish news not found")
        return data
    except Exception as e:
        logger.error(f"Error in bullish news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/bearish")
@debug_endpoint
async def get_news_bearish(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار نزولی"""
    try:
        data = coin_stats_manager.get_news_by_type("bearish", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Bearish news not found")
        return data
    except Exception as e:
        logger.error(f"Error in bearish news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/{news_id}")
@debug_endpoint
async def get_news_detail(news_id: str):
    """دریافت جزئیات خبر"""
    try:
        data = coin_stats_manager.get_news_detail(news_id)
        if not data:
            raise HTTPException(status_code=404, detail="News not found")
        return data
    except Exception as e:
        logger.error(f"Error in news detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های بینش بازار ====================

@router.get("/insights/btc-dominance")
@debug_endpoint
async def get_btc_dominance(
    period_type: str = Query("all", regex="^(all|24h|1w|1m|3m|1y)$")
):
    """دریافت دامیننس بیت‌کوین"""
    try:
        data = coin_stats_manager.get_btc_dominance(period_type)
        if not data:
            raise HTTPException(status_code=404, detail="BTC dominance data not found")
        return data
    except Exception as e:
        logger.error(f"Error in BTC dominance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/fear-greed")
@debug_endpoint
async def get_fear_greed():
    """دریافت شاخص ترس و طمع"""
    try:
        data = coin_stats_manager.get_fear_greed()
        if not data:
            raise HTTPException(status_code=404, detail="Fear & greed data not found")
        return data
    except Exception as e:
        logger.error(f"Error in fear greed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/fear-greed/chart")
@debug_endpoint
async def get_fear_greed_chart():
    """دریافت چارت ترس و طمع"""
    try:
        data = coin_stats_manager.get_fear_greed_chart()
        if not data:
            raise HTTPException(status_code=404, detail="Fear & greed chart not found")
        return data
    except Exception as e:
        logger.error(f"Error in fear greed chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/rainbow-chart/{coin_id}")
@debug_endpoint
async def get_rainbow_chart(coin_id: str):
    """دریافت چارت رنگین‌کمان"""
    try:
        data = coin_stats_manager.get_rainbow_chart(coin_id)
        if not data:
            raise HTTPException(status_code=404, detail="Rainbow chart not found")
        return data
    except Exception as e:
        logger.error(f"Error in rainbow chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های WebSocket ====================

@router.get("/websocket/status")
@debug_endpoint
async def websocket_status():
    """وضعیت WebSocket"""
    if not lbank_ws:
        raise HTTPException(status_code=503, detail="WebSocket not initialized")
    
    return lbank_ws.get_connection_status()

@router.get("/websocket/data/{symbol}")
@debug_endpoint
async def get_websocket_data(symbol: str):
    """داده‌های لحظه‌ای WebSocket"""
    if not lbank_ws:
        raise HTTPException(status_code=503, detail="WebSocket not initialized")
    
    data = lbank_ws.get_realtime_data(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found in WebSocket data")
    return data

@router.get("/websocket/pairs/active")
@debug_endpoint
async def get_active_pairs():
    """لیست جفت ارزهای فعال"""
    if not lbank_ws:
        raise HTTPException(status_code=503, detail="WebSocket not initialized")
    
    return {
        "active_pairs": lbank_ws.get_active_pairs(),
        "total": len(lbank_ws.get_active_pairs())
    }

# ==================== روت‌های سیستم و دیباگ ====================

@router.get("/system/debug")
async def get_debug_info():
    """اطلاعات دیباگ کامل"""
    try:
        return debug_manager.generate_debug_report()
    except Exception as e:
        logger.error(f"Error generating debug report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/cache/info")
async def get_cache_info():
    """اطلاعات کش"""
    try:
        return coin_stats_manager.get_cache_info()
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/cache/clear")
async def clear_cache(endpoint: str = None):
    """پاک کردن کش"""
    try:
        coin_stats_manager.clear_cache(endpoint)
        return {"status": "success", "message": f"Cache cleared for {endpoint or 'all endpoints'}"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health")
async def system_health():
    """سلامت سیستم"""
    try:
        system_health = debug_manager.get_system_health()
        api_stats = debug_manager.get_api_stats()
        
        return {
            "system": system_health,
            "api": api_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های هشدار ====================

@router.post("/alerts/create")
@debug_endpoint
async def create_alert(request: AlertRequest):
    """ایجاد هشدار"""
    try:
        alert_id = f"alert_{int(datetime.now().timestamp())}"
        
        alert_data = {
            "id": alert_id,
            "symbol": request.symbol.upper(),
            "condition": request.condition,
            "target_price": request.target_price,
            "alert_type": request.alert_type,
            "status": "ACTIVE",
            "created_at": datetime.now().isoformat()
        }
        
        alerts_db[alert_id] = alert_data
        
        return {
            "alert_id": alert_id,
            "status": "SUCCESS",
            "message": f"Alert created for {request.symbol}",
            "alert": alert_data
        }
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/list")
@debug_endpoint
async def list_alerts():
    """لیست هشدارها"""
    return {
        "alerts": list(alerts_db.values()),
        "total_count": len(alerts_db)
    }

@router.delete("/alerts/{alert_id}")
@debug_endpoint
async def delete_alert(alert_id: str):
    """حذف هشدار"""
    if alert_id in alerts_db:
        deleted_alert = alerts_db.pop(alert_id)
        return {
            "status": "SUCCESS",
            "message": f"Alert {alert_id} deleted",
            "deleted_alert": deleted_alert
        }
    else:
        raise HTTPException(status_code=404, detail="Alert not found")
