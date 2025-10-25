# complete_routes.py - نسخه کامل و اصلاح شده
from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import requests
import logging

# تنظیمات API
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': 'oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=',
    'timeout': 30
}

logger = logging.getLogger(__name__)
router = APIRouter()

# مدل‌های درخواست
class AlertRequest(BaseModel):
    symbol: str
    condition: str
    target_price: float
    alert_type: str = "price"

alerts_db = {}

# سرویس داده‌های خارجی
class ExternalDataService:
    def __init__(self):
        self.api_base_url = API_CONFIG['base_url']
        self.api_key = API_CONFIG['api_key']
        self.headers = {"X-API-KEY": self.api_key}
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """ساخت درخواست به API خارجی"""
        url = f"{self.api_base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"API request error to {endpoint}: {e}")
            return {}

    # ==================== اندپوینت‌های کوین‌ها ====================
    
    def get_coins_list(self, limit: int = 20, page: int = 1, currency: str = "USD", 
                      sort_by: str = "rank", sort_dir: str = "asc") -> Dict:
        params = {
            "limit": limit,
            "page": page,
            "currency": currency,
            "sortBy": sort_by,
            "sortDir": sort_dir
        }
        return self._make_api_request("coins", params)

    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        params = {"currency": currency}
        return self._make_api_request(f"coins/{coin_id}", params)

    def get_coin_charts(self, coin_id: str, period: str = "1w") -> Dict:
        valid_periods = ["24h", "1w", "1m", "3m", "6m", "1y", "all"]
        if period not in valid_periods:
            period = "1w"
        params = {"period": period}
        return self._make_api_request(f"coins/{coin_id}/charts", params)

    def get_coins_charts(self, coin_ids: str, period: str = "1w") -> Dict:
        valid_periods = ["24h", "1w", "1m", "3m", "6m", "1y", "all"]
        if period not in valid_periods:
            period = "1w"
        params = {
            "coinIds": coin_ids,
            "period": period
        }
        return self._make_api_request("coins/charts", params)

    def get_coin_price_avg(self, coin_id: str, timestamp: str) -> Dict:
        params = {
            "coinId": coin_id,
            "timestamp": timestamp
        }
        return self._make_api_request("coins/price/avg", params)

    def get_exchange_price(self, exchange: str, from_coin: str, to_coin: str, timestamp: str) -> Dict:
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        return self._make_api_request("coins/price/exchange", params)

    # ==================== اندپوینت‌های بازار و صرافی ====================
    
    def get_tickers_exchanges(self) -> Dict:
        return self._make_api_request("tickers/exchanges")

    def get_tickers_markets(self) -> Dict:
        return self._make_api_request("tickers/markets")

    def get_markets(self) -> Dict:
        return self._make_api_request("markets")

    def get_fiats(self) -> Dict:
        return self._make_api_request("fiats")

    def get_currencies(self) -> Dict:
        return self._make_api_request("currencies")

    # ==================== اندپوینت‌های اخبار ====================
    
    def get_news_sources(self) -> Dict:
        return self._make_api_request("news/sources")

    def get_news(self, limit: int = 50) -> Dict:
        params = {"limit": limit}
        return self._make_api_request("news", params)

    def get_news_by_type(self, news_type: str, limit: int = 50) -> Dict:
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            news_type = "latest"
        return self._make_api_request(f"news/type/{news_type}", {"limit": limit})

    def get_news_detail(self, news_id: str) -> Dict:
        return self._make_api_request(f"news/{news_id}")

    # ==================== اندپوینت‌های بینش بازار ====================
    
    def get_btc_dominance(self, period_type: str = "all") -> Dict:
        valid_periods = ["all", "24h", "1w", "1m", "3m", "1y"]
        if period_type not in valid_periods:
            period_type = "all"
        params = {"type": period_type}
        return self._make_api_request("insights/btc-dominance", params)

    def get_fear_greed(self) -> Dict:
        return self._make_api_request("insights/fear-and-greed")

    def get_fear_greed_chart(self) -> Dict:
        return self._make_api_request("insights/fear-and-greed/chart")

    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Dict:
        return self._make_api_request(f"insights/rainbow-chart/{coin_id}")

# ایجاد سرویس
external_service = ExternalDataService()

# WebSocket manager
class DummyWebSocket:
    def __init__(self):
        self.connected = False
        self.realtime_data = {}
    
    def get_realtime_data(self, symbol: str):
        return self.realtime_data.get(symbol, {})

lbank_ws = DummyWebSocket()

def set_websocket_manager(ws_manager):
    global lbank_ws
    lbank_ws = ws_manager

# ==================== روت‌های اصلی ====================

@router.get("/")
async def root():
    return {
        "message": "AI Trading Assistant API - Complete Version", 
        "version": "4.0.0",
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
            "alerts": {
                "create": "/alerts/create",
                "list": "/alerts/list",
                "delete": "/alerts/{alert_id}"
            }
        }
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "external_api": "connected",
            "websocket": "connected" if lbank_ws.connected else "disconnected"
        }
    }

# ==================== روت‌های کوین‌ها ====================

@router.get("/coins/list")
async def get_coins_list(
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    currency: str = Query("USD"),
    sort_by: str = Query("rank", regex="^(rank|marketCap|price|volume|name|symbol)$"),
    sort_dir: str = Query("asc", regex="^(asc|desc)$")
):
    try:
        data = external_service.get_coins_list(
            limit=limit, 
            page=page, 
            currency=currency,
            sort_by=sort_by,
            sort_dir=sort_dir
        )
        
        if not data:
            raise HTTPException(status_code=404, detail="No data received")
            
        return {
            "data": data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": len(data.get('result', []))
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching coins: {str(e)}")

@router.get("/coins/{coin_id}")
async def get_coin_details(
    coin_id: str,
    currency: str = Query("USD")
):
    try:
        data = external_service.get_coin_details(coin_id, currency)
        if not data:
            raise HTTPException(status_code=404, detail="Coin not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/{coin_id}/charts")
async def get_coin_charts(
    coin_id: str,
    period: str = Query("1w", regex="^(24h|1w|1m|3m|6m|1y|all)$")
):
    try:
        data = external_service.get_coin_charts(coin_id, period)
        if not data:
            raise HTTPException(status_code=404, detail="Chart data not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/charts/multi")
async def get_multi_coin_charts(
    coin_ids: str = Query(..., alias="coinIds", description="لیست کوین‌ها (مثلاً bitcoin,ethereum,solana)"),
    period: str = Query("1w", regex="^(24h|1w|1m|3m|6m|1y|all)$")
):
    try:
        data = external_service.get_coins_charts(coin_ids, period)
        if not data:
            raise HTTPException(status_code=404, detail="Chart data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/price/avg")
async def get_coin_price_avg(
    coin_id: str = Query(..., alias="coinId", description="شناسه کوین"),
    timestamp: str = Query(..., description="تایم‌استمپ")
):
    try:
        data = external_service.get_coin_price_avg(coin_id, timestamp)
        if not data:
            raise HTTPException(status_code=404, detail="Price data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/price/exchange")
async def get_exchange_price(
    exchange: str = Query(..., description="نام صرافی"),
    from_coin: str = Query(..., alias="from", description="ارز مبدأ"),
    to_coin: str = Query(..., alias="to", description="ارز مقصد"),
    timestamp: str = Query(..., description="تایم‌استمپ")
):
    try:
        data = external_service.get_exchange_price(exchange, from_coin, to_coin, timestamp)
        if not data:
            raise HTTPException(status_code=404, detail="Exchange price not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های بازار و صرافی ====================

@router.get("/market/overview")
async def market_overview():
    try:
        top_coins = external_service.get_coins_list(limit=10)
        btc_dominance = external_service.get_btc_dominance()
        fear_greed = external_service.get_fear_greed()
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "market_data": {
                "top_coins": top_coins.get('result', [])[:5] if top_coins else [],
                "total_coins": len(top_coins.get('result', [])) if top_coins else 0
            }
        }
        
        if btc_dominance:
            response["btc_dominance"] = btc_dominance
        if fear_greed:
            response["fear_greed"] = fear_greed
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/exchanges")
async def get_exchanges():
    try:
        data = external_service.get_tickers_exchanges()
        if not data:
            raise HTTPException(status_code=404, detail="Exchanges data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/markets")
async def get_markets():
    try:
        data = external_service.get_tickers_markets()
        if not data:
            data = external_service.get_markets()
        if not data:
            raise HTTPException(status_code=404, detail="Markets data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/fiats")
async def get_fiats():
    try:
        data = external_service.get_fiats()
        if not data:
            raise HTTPException(status_code=404, detail="Fiats data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/currencies")
async def get_currencies():
    try:
        data = external_service.get_currencies()
        if not data:
            raise HTTPException(status_code=404, detail="Currencies data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های اخبار ====================

@router.get("/news/sources")
async def get_news_sources():
    try:
        data = external_service.get_news_sources()
        if not data:
            raise HTTPException(status_code=404, detail="News sources not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news")
async def get_news(limit: int = Query(50, ge=1, le=100)):
    try:
        data = external_service.get_news(limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="News not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/handpicked")
async def get_news_handpicked(limit: int = Query(50, ge=1, le=100)):
    try:
        data = external_service.get_news_by_type("handpicked", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Handpicked news not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/trending")
async def get_news_trending(limit: int = Query(50, ge=1, le=100)):
    try:
        data = external_service.get_news_by_type("trending", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Trending news not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/latest")
async def get_news_latest(limit: int = Query(50, ge=1, le=100)):
    try:
        data = external_service.get_news_by_type("latest", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Latest news not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/bullish")
async def get_news_bullish(limit: int = Query(50, ge=1, le=100)):
    try:
        data = external_service.get_news_by_type("bullish", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Bullish news not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/bearish")
async def get_news_bearish(limit: int = Query(50, ge=1, le=100)):
    try:
        data = external_service.get_news_by_type("bearish", limit)
        if not data:
            raise HTTPException(status_code=404, detail="Bearish news not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/{news_id}")
async def get_news_detail(news_id: str):
    try:
        data = external_service.get_news_detail(news_id)
        if not data:
            raise HTTPException(status_code=404, detail="News not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های بینش بازار ====================

@router.get("/insights/btc-dominance")
async def get_btc_dominance(
    period_type: str = Query("all", regex="^(all|24h|1w|1m|3m|1y)$")
):
    try:
        data = external_service.get_btc_dominance(period_type)
        if not data:
            raise HTTPException(status_code=404, detail="BTC dominance data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/fear-greed")
async def get_fear_greed():
    try:
        data = external_service.get_fear_greed()
        if not data:
            raise HTTPException(status_code=404, detail="Fear & greed data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/fear-greed/chart")
async def get_fear_greed_chart():
    try:
        data = external_service.get_fear_greed_chart()
        if not data:
            raise HTTPException(status_code=404, detail="Fear & greed chart not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/rainbow-chart/{coin_id}")
async def get_rainbow_chart(coin_id: str):
    try:
        data = external_service.get_rainbow_chart(coin_id)
        if not data:
            raise HTTPException(status_code=404, detail="Rainbow chart not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های هشدار ====================

@router.post("/alerts/create")
async def create_alert(request: AlertRequest):
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
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/list")
async def list_alerts():
    return {
        "alerts": list(alerts_db.values()),
        "total_count": len(alerts_db),
        "timestamp": datetime.now().isoformat()
    }

@router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    if alert_id in alerts_db:
        deleted_alert = alerts_db.pop(alert_id)
        return {
            "status": "SUCCESS",
            "message": f"Alert {alert_id} deleted",
            "deleted_alert": deleted_alert
        }
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@router.get("/system/status")
async def system_status():
    return {
        "timestamp": datetime.now().isoformat(),
        "external_api": "connected",
        "websocket": "connected" if lbank_ws.connected else "disconnected",
        "active_alerts": len(alerts_db),
        "version": "4.0.0"
    }
