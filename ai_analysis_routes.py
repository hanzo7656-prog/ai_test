# ai_analysis_routes.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
import json
import os
import glob
from datetime import datetime
import requests

router = APIRouter(prefix="/ai", tags=["AI Analysis"])

class AIAnalysisService:
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
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error to {endpoint}: {e}")
            return {}
    
    def get_coin_data(self, symbol: str, currency: str = "USD") -> Dict[str, Any]:
        """دریافت داده‌های کامل یک کوین"""
        # اول از داده‌های خام
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام
        for filename, data in raw_data.items():
            if symbol.lower() in filename.lower():
                print(f"Found raw data for {symbol}: {filename}")
                return data
        
        # اگر پیدا نشد از API استفاده کن
        coin_data = self._make_api_request(f"coins/{symbol}", {"currency": currency})
        return coin_data.get('result', {}) if 'result' in coin_data else coin_data
    
    def get_historical_data(self, symbol: str, period: str = "all") -> Dict[str, Any]:
        """دریافت داده‌های تاریخی"""
        return self._make_api_request(f"coins/{symbol}/charts", {"period": period})
    
    def get_market_insights(self) -> Dict[str, Any]:
        """دریافت بینش‌های بازار"""
        insights = {}
        
        # ترس و طمع
        fear_greed = self._make_api_request("insights/fear-and-greed")
        if fear_greed:
            insights["fear_greed"] = fear_greed
        
        # دامیننس بیت‌کوین
        btc_dominance = self._make_api_request("insights/btc-dominance", {"type": "all"})
        if btc_dominance:
            insights["btc_dominance"] = btc_dominance
        
        return insights
    
    def get_news_data(self, limit: int = 10) -> Dict[str, Any]:
        """دریافت داده‌های اخبار"""
        news_data = {}
        
        # اخبار عمومی
        general_news = self._make_api_request("news", {"limit": limit})
        if general_news:
            news_data["general"] = general_news
        
        return news_data
    
    def get_market_data(self) -> Dict[str, Any]:
        """دریافت داده‌های بازار"""
        market_data = {}
        
        # لیست کوین‌ها با اطلاعات بازار
        coins_list = self._make_api_request("coins", {"limit": 50})
        if coins_list and 'result' in coins_list:
            market_data["top_coins"] = coins_list["result"]
        
        return market_data
    
    def prepare_ai_input(self, symbols: List[str], period: str = "7d") -> Dict[str, Any]:
        """آماده‌سازی داده‌های ورودی برای هوش مصنوعی"""
        
        ai_input = {
            "timestamp": int(datetime.now().timestamp()),
            "analysis_scope": "multi_symbol" if len(symbols) > 1 else "single_symbol",
            "period": period,
            "symbols": symbols,
            "data_sources": {
                "repo_data": False,
                "api_data": False
            },
            "market_data": {},
            "symbols_data": {},
            "news_data": {},
            "insights_data": {}
        }
        
        # بارگذاری داده‌های خام
        raw_data = self._load_raw_data()
        if raw_data:
            ai_input["data_sources"]["repo_data"] = True
            ai_input["raw_files_count"] = len(raw_data)
        
        # داده‌های بازار
        market_data = self.get_market_data()
        if market_data:
            ai_input["market_data"] = market_data
            ai_input["data_sources"]["api_data"] = True
        
        # بینش‌های بازار
        insights = self.get_market_insights()
        if insights:
            ai_input["insights_data"] = insights
        
        # اخبار
        news = self.get_news_data()
        if news:
            ai_input["news_data"] = news
        
        # داده‌های هر نماد
        for symbol in symbols:
            symbol_data = {}
            
            # اطلاعات اصلی کوین
            coin_data = self.get_coin_data(symbol)
            if coin_data:
                symbol_data["coin_info"] = coin_data
            
            # داده‌های تاریخی
            historical_data = self.get_historical_data(symbol, period)
            if historical_data:
                symbol_data["historical"] = historical_data
            
            if symbol_data:
                ai_input["symbols_data"][symbol] = symbol_data
        
        return ai_input

# ایجاد سرویس
ai_service = AIAnalysisService()

@router.post("/analysis")
async def ai_analysis(
    symbols: List[str] = Query(..., description="نمادها برای تحلیل"),
    period: str = Query("7d", regex="^(1h|4h|1d|7d|30d|90d|all)$"),
    include_news: bool = True,
    include_market_data: bool = True
):
    """
    تحلیل هوش مصنوعی با داده‌های واقعی از ریپو و API
    """
    try:
        # آماده‌سازی داده‌های ورودی برای AI
        ai_input = ai_service.prepare_ai_input(symbols, period)
        
        # اگر داده‌ای دریافت نشد
        if not ai_input["data_sources"]["repo_data"] and not ai_input["data_sources"]["api_data"]:
            raise HTTPException(
                status_code=503, 
                detail="هیچ منبع داده‌ای در دسترس نیست"
            )
        
        return {
            "status": "success",
            "message": "داده‌ها برای پردازش AI آماده شدند",
            "analysis_id": f"ai_analysis_{int(datetime.now().timestamp())}",
            "data_summary": {
                "symbols_processed": len(ai_input["symbols_data"]),
                "market_data_available": bool(ai_input["market_data"]),
                "news_data_available": bool(ai_input["news_data"]),
                "insights_available": bool(ai_input["insights_data"]),
                "data_sources": ai_input["data_sources"]
            },
            "ai_input_data": ai_input
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل: {str(e)}")

@router.get("/analysis/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """دریافت وضعیت تحلیل"""
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "timestamp": int(datetime.now().timestamp())
    }
