# complete_routes.py - نسخه کاملاً اصلاح شده
from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os
import glob
from datetime import datetime
import requests
import logging

# تنظیمات API - مستقیم از config کپی شده
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': 'oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=',
    'timeout': 30
}

# تنظیم لاگینگ
logger = logging.getLogger(__name__)

router = APIRouter()

# مدل‌های درخواست
class AlertRequest(BaseModel):
    symbol: str
    condition: str
    target_price: float
    alert_type: str = "price"

# دیتابیس ساده برای هشدارها
alerts_db = {}

# سرویس مستقل برای جلوگیری از circular import
class DataService:
    def __init__(self):
        self.api_base_url = API_CONFIG['base_url']
        self.api_key = API_CONFIG['api_key']
        self.headers = {"X-API-KEY": self.api_key}
        self.raw_data_path = "./raw_data"
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """ساخت درخواست به API"""
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
    
    def get_coins_list(self, limit: int = 50) -> List[Dict]:
        """دریافت لیست کوین‌ها - نسخه تعمیر شده"""
        try:
            params = {"limit": limit, "currency": "USD"}
            coins_data = self._make_api_request("coins", params)
            
            if coins_data and 'result' in coins_data:
                return coins_data['result']
            else:
                logger.warning("No data received from API")
                return []
                
        except Exception as e:
            logger.error(f"Error getting coins list: {e}")
            return []

    def get_coin_details(self, coin_id: str) -> Dict:
        """دریافت جزئیات کوین خاص"""
        return self._make_api_request(f"coins/{coin_id}")

    def get_coin_charts(self, coin_id: str, period: str = "1w") -> Dict:
        """دریافت چارت کوین"""
        params = {"period": period}
        return self._make_api_request(f"coins/{coin_id}/charts", params)

# ایجاد سرویس داده
data_service = DataService()

# WebSocket manager - مقدار پیش‌فرض برای جلوگیری از خطا
class DummyWebSocket:
    def __init__(self):
        self.connected = False
        self.realtime_data = {}
    
    def get_realtime_data(self, symbol: str):
        return self.realtime_data.get(symbol, {})

lbank_ws = DummyWebSocket()

def set_websocket_manager(ws_manager):
    """تنظیم WebSocket manager از run.py"""
    global lbank_ws
    lbank_ws = ws_manager

# ==================== روت‌های اصلی ====================

@router.get("/")
async def root():
    """صفحه اصلی"""
    return {
        "message": "AI Trading Assistant API", 
        "version": "3.0.0",
        "endpoints": [
            "/health",
            "/coins/list",
            "/coins/{coin_id}",
            "/symbols/list",
            "/market/overview"
        ]
    }

@router.get("/health")
async def health_check():
    """سلامت سرویس"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "websocket": "connected" if lbank_ws and lbank_ws.connected else "disconnected"
        }
    }

# ==================== روت‌های کوین‌ها ====================

@router.get("/coins/list")
async def get_coins_list(
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1)
):
    """دریافت لیست کوین‌ها"""
    try:
        coins = data_service.get_coins_list(limit=limit)
        
        return {
            "coins": coins,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": len(coins)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching coins: {str(e)}")

@router.get("/coins/{coin_id}")
async def get_coin_details(coin_id: str):
    """دریافت جزئیات کوین خاص"""
    try:
        coin_data = data_service.get_coin_details(coin_id)
        if not coin_data:
            raise HTTPException(status_code=404, detail="Coin not found")
        return coin_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coins/{coin_id}/charts")
async def get_coin_charts(
    coin_id: str,
    period: str = Query("1w", regex="^(24h|1w|1m|3m|6m|1y|all)$")
):
    """دریافت چارت کوین"""
    try:
        chart_data = data_service.get_coin_charts(coin_id, period)
        if not chart_data:
            raise HTTPException(status_code=404, detail="Chart data not found")
        return chart_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت‌های بازار ====================

@router.get("/market/overview")
async def market_overview():
    """نمای کلی بازار"""
    try:
        # دریافت ۱۰ کوین برتر
        top_coins = data_service.get_coins_list(limit=10)
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "market_summary": {
                "total_coins": len(top_coins),
                "top_performers": []
            }
        }
        
        if top_coins:
            for coin in top_coins[:5]:  # ۵ کوین برتر
                response["market_summary"]["top_performers"].append({
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol"),
                    "price": coin.get("price"),
                    "change_24h": coin.get("priceChange1d", 0)
                })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols/list")
async def list_symbols():
    """لیست کامل نمادها"""
    try:
        coins = data_service.get_coins_list(limit=100)
        symbols = []
        
        for coin in coins:
            if 'symbol' in coin:
                symbols.append({
                    "symbol": coin["symbol"],
                    "name": coin.get("name", ""),
                    "price": coin.get("price", 0)
                })
        
        # اگر داده‌ای نبود، لیست پیش‌فرض
        if not symbols:
            symbols = [
                {"symbol": "BTC", "name": "Bitcoin", "price": 0},
                {"symbol": "ETH", "name": "Ethereum", "price": 0},
                {"symbol": "SOL", "name": "Solana", "price": 0}
            ]
        
        return {
            "symbols": symbols[:50],  # حداکثر ۵۰ نماد
            "total_symbols": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in list_symbols: {e}")
        return {
            "symbols": [
                {"symbol": "BTC", "name": "Bitcoin", "price": 0},
                {"symbol": "ETH", "name": "Ethereum", "price": 0}
            ],
            "total_symbols": 2,
            "error": "Using fallback data"
        }

# ==================== روت‌های هشدار ====================

@router.post("/alerts/create")
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
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/list")
async def list_alerts():
    """لیست هشدارها"""
    return {
        "alerts": list(alerts_db.values()),
        "total_count": len(alerts_db)
    }
