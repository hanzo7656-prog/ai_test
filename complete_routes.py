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
        self.api_base_url = "https://openapiv1.coinstats.app"
        self.api_key = "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="
        self.headers = {"X-API-KEY": self.api_key}
        self.raw_data_path = "./raw_data"
    
    def _load_raw_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های خام از ریپو"""
        raw_data = {}
        try:
            for folder in ["A", "B", "C", "D"]:
                folder_path = os.path.join(self.raw_data_path, folder)
                if os.path.exists(folder_path):
                    data_files = glob.glob(f"{folder_path}/**/*.json", recursive=True)
                    for file_path in data_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                filename = os.path.basename(file_path)
                                raw_data[filename] = json.load(f)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        except Exception as e:
            print(f"Error in raw data loading: {e}")
        return raw_data
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """ساخت درخواست به API"""
        url = f"{self.api_base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error to {endpoint}: {e}")
            return {}
    
    def get_news_data(self, limit: int = 10) -> Dict[str, Any]:
        """دریافت داده‌های اخبار"""
        news_data = {}
        
        # اخبار عمومی
        general_news = self._make_api_request("news", {"limit": limit})
        if general_news:
            news_data["general"] = general_news
        
        return news_data
    
    def get_market_insights(self) -> Dict[str, Any]:
        """دریافت بینش‌های بازار"""
        insights = {}
        
        # ترس و طمع
        fear_greed = self._make_api_request("insights/fear-and-greed")
        if fear_greed:
            insights["fear_greed"] = fear_greed
        
        return insights

    def get_coins_list(self, limit: int = 150) -> List[Dict]:
        """دریافت لیست کوین‌ها - نسخه تعمیر شده"""
        try:
            coins_data = self._make_api_request("coins", {"limit": limit})
            if coins_data and 'result' in coins_data:
                return coins_data['result']
            return []
        except Exception as e:
            logger.error(f"Error getting coins list: {e}")
            return []

# ایجاد سرویس داده
data_service = DataService()

# WebSocket manager - این باید از run.py inject بشه
lbank_ws = None

def set_websocket_manager(ws_manager):
    """تنظیم WebSocket manager از run.py"""
    global lbank_ws
    lbank_ws = ws_manager

# ==================== روت‌های اصلی ====================

@router.get("/")
async def root():
    """صفحه اصلی"""
    return {"message": "AI Trading Assistant API", "version": "3.0.0"}

@router.get("/health")
async def health_check():
    """سلامت سرویس"""
    return {
        "status": "healthy",
        "timestamp": int(datetime.now().timestamp()),
        "services": {
            "api": "running",
            "websocket": "connected" if lbank_ws and lbank_ws.connected else "disconnected",
            "data_service": "ready"
        }
    }

@router.get("/websocket/status")
async def websocket_status():
    """وضعیت WebSocket"""
    if not lbank_ws:
        raise HTTPException(status_code=503, detail="WebSocket not initialized")
    
    return {
        "connected": lbank_ws.connected,
        "subscribed_pairs": list(lbank_ws.realtime_data.keys()),
        "data_count": len(lbank_ws.realtime_data)
    }

@router.get("/websocket/data/{symbol}")
async def get_websocket_data(symbol: str):
    """داده‌های لحظه‌ای WebSocket"""
    if not lbank_ws:
        raise HTTPException(status_code=503, detail="WebSocket not initialized")
    
    data = lbank_ws.get_realtime_data(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found in WebSocket data")
    return data

# ==================== روت‌های اخبار ====================

@router.get("/news/latest")
async def get_latest_news(
    limit: int = Query(20, ge=1, le=100)
):
    """آخرین اخبار"""
    try:
        news_data = data_service.get_news_data(limit)
        return {
            "news": news_data,
            "count": len(news_data.get('general', [])),
            "timestamp": int(datetime.now().timestamp())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

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
            "created_at": int(datetime.now().timestamp()),
            "triggered": False
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
async def list_alerts(symbol: Optional[str] = None):
    """لیست هشدارها"""
    alerts = list(alerts_db.values())
    
    if symbol:
        alerts = [alert for alert in alerts if alert["symbol"] == symbol.upper()]
    
    return {
        "alerts": alerts,
        "total_count": len(alerts),
        "active_count": len([a for a in alerts if a["status"] == "ACTIVE"])
    }

@router.post("/alerts/{alert_id}/delete")
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

# ==================== روت‌های داده‌های خام ====================

@router.get("/data/raw/{data_type}")
async def get_raw_data(
    data_type: str = Path(..., regex="^(coins|charts|news|market|insights)$"),
    symbol: str = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """داده‌های خام از ریپو"""
    try:
        raw_data = data_service._load_raw_data()
        
        filtered_data = {}
        for filename, data in raw_data.items():
            if data_type in filename.lower():
                if symbol and symbol.lower() not in filename.lower():
                    continue
                filtered_data[filename] = data
        
        return {
            "data_type": data_type,
            "symbol": symbol,
            "files_count": len(filtered_data),
            "data": filtered_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/sources")
async def list_data_sources():
    """لیست منابع داده"""
    return {
        "repo_sources": [
            "https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data/A",
            "https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data/B",
            "https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data/C", 
            "https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data/D"
        ],
        "api_sources": [
            "CoinStats REST API",
            "LBank WebSocket"
        ]
    }

# ==================== روت‌های مدیریتی ====================

@router.get("/system/performance")
async def system_performance():
    """کارایی سیستم"""
    return {
        "timestamp": int(datetime.now().timestamp()),
        "websocket_status": {
            "connected": lbank_ws.connected if lbank_ws else False,
            "active_pairs": len(lbank_ws.realtime_data) if lbank_ws else 0
        },
        "api_status": "healthy",
        "memory_usage": "normal"
    }

@router.get("/market/overview")
async def market_overview():
    """نمای کلی بازار"""
    try:
        # داده‌های واقعی از CoinStats
        coins_data = data_service._make_api_request("coins", {"limit": 10})
        market_insights = data_service.get_market_insights()
        
        response = {
            "timestamp": int(datetime.now().timestamp()),
            "market_insights": market_insights
        }
        
        if coins_data and 'result' in coins_data:
            response["top_coins"] = coins_data['result'][:5]
        
        if lbank_ws:
            response["websocket_data"] = {
                "btc_price": lbank_ws.get_realtime_data('btc_usdt').get('price', 0),
                "eth_price": lbank_ws.get_realtime_data('eth_usdt').get('price', 0)
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols/list")
async def list_symbols():
    """لیست کامل نمادها - نسخه تعمیر شده"""
    try:
        # دریافت لیست کامل کوین‌ها
        coins_data = data_service.get_coins_list(limit=150)
        all_symbols = []
        
        if coins_data:
            for coin in coins_data:
                if 'symbol' in coin:
                    all_symbols.append(coin['symbol'])
        
        # اگر از API چیزی نگرفتیم، لیست پیش‌فرض
        if not all_symbols:
            all_symbols = [
                "BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOT", "LTC", "LINK", "MATIC",
                "AVAX", "DOGE", "ATOM", "XLM", "ALGO", "NEAR", "FTM", "SAND", "MANA", "UNI"
            ]
        
        # نمادهای WebSocket
        websocket_symbols = []
        if lbank_ws and hasattr(lbank_ws, 'realtime_data'):
            websocket_symbols = list(lbank_ws.realtime_data.keys())
        
        return {
            "symbols": all_symbols[:50],  # حداکثر ۵۰ نماد
            "websocket_symbols": websocket_symbols,
            "total_symbols": len(all_symbols),
            "active_websocket_pairs": len(websocket_symbols),
            "data_source": "coinstats_api"
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در دریافت لیست نمادها: {e}")
        return {
            "symbols": ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOT", "LTC"],
            "websocket_symbols": [],
            "total_symbols": 8,
            "active_websocket_pairs": 0,
            "error": str(e)
        }
