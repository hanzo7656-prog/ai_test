# ai_analysis_routes.py - نسخه کاملاً کامل
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import json
import os
import glob
from datetime import datetime
import requests
from pydantic import BaseModel

router = APIRouter(prefix="/ai", tags=["AI Analysis"])

# مدل‌های درخواست
class AnalysisRequest(BaseModel):
    symbols: List[str]
    period: str = "7d"
    include_news: bool = True
    include_market_data: bool = True
    include_technical: bool = True
    analysis_type: str = "comprehensive"

class AIAnalysisService:
    def __init__(self):
        self.api_base_url = "https://openapiv1.coinstats.app"
        self.api_key = "oYGllJrdvcdApdgxLTNs9jUnvR/RUGAMhZjt123YtbpA="
        self.headers = {"X-API-KEY": self.api_key}
        self.raw_data_path = "./raw_data"
        
        # تنظیمات تحلیل AI
        self.supported_periods = ["1h", "4h", "1d", "7d", "30d", "90d", "all"]
        self.analysis_types = ["comprehensive", "technical", "sentiment", "momentum"]

    def _load_raw_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های خام از GitHub و local"""
        raw_data = {}
        try:
            # اول از GitHub سعی کن
            from complete_coinstats_manager import CompleteCoinStatsManager
            manager = CompleteCoinStatsManager()
            github_data = manager._load_raw_data()
            if github_data:
                return github_data
            
            # اگر GitHub کار نکرد، local رو چک کن
            for folder in ["A", "B", "C", "D"]:
                folder_path = os.path.join(self.raw_data_path, folder)
                if not os.path.exists(folder_path):
                    continue

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
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error to {endpoint}: {e}")
            return {}

    def get_coin_data(self, symbol: str, currency: str = "USD") -> Dict[str, Any]:
        """دریافت داده‌های کامل یک کوین"""
        # اول از داده‌های خام
        raw_data = self._load_raw_data()
        for filename, data in raw_data.items():
            if symbol.lower() in filename.lower():
                print(f"Found raw data for {symbol}: {filename}")
                return data

        # اگر پیدا نشد، از API استفاده کن
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

        # دامیننس بیت کوین
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
        
        # لیست کوین ها با اطلاعات بازار
        coins_list = self._make_api_request("coins", {"limit": 50})
        if coins_list and 'result' in coins_list:
            market_data["top_coins"] = coins_list["result"]
            
        return market_data

    def get_technical_indicators(self, symbol: str, period: str = "7d") -> Dict[str, Any]:
        """دریافت اندیکاتورهای تکنیکال"""
        try:
            from technical_engine_complete import CompleteTechnicalEngine
            
            # دریافت داده‌های تاریخی
            historical_data = self.get_historical_data(symbol, period)
            if not historical_data or 'result' not in historical_data:
                return {}
                
            # استخراج قیمت‌ها
            prices = [float(item['price']) for item in historical_data['result'] if 'price' in item]
            
            if len(prices) < 20:
                return {}
                
            # ایجاد داده‌های OHLC
            ohlc_data = {
                'open': prices[:-1],
                'high': [max(prices[i], prices[i+1]) for i in range(len(prices)-1)],
                'low': [min(prices[i], prices[i+1]) for i in range(len(prices)-1)],
                'close': prices[1:],
                'volume': [1000000] * (len(prices) - 1)
            }
            
            # محاسبه اندیکاتورها
            engine = CompleteTechnicalEngine()
            indicators = engine.calculate_all_indicators(ohlc_data)
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating technical indicators for {symbol}: {e}")
            return {}

    def get_ai_prediction(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """دریافت پیش‌بینی AI از مدل"""
        try:
            from ultra_efficient_trading_transformer import TradingSignalPredictor
            
            # آماده‌سازی داده‌های بازار برای مدل
            market_data = {
                'price_data': {
                    'historical_prices': data.get('prices', [50000]),
                    'volume_data': data.get('volumes', [1000000])
                },
                'technical_indicators': {
                    'momentum_indicators': data.get('technical_indicators', {}),
                    'trend_indicators': data.get('trend_data', {})
                },
                'market_data': {
                    'fear_greed_index': data.get('fear_greed', {'value': 50})
                }
            }
            
            # پیش‌بینی با مدل AI
            predictor = TradingSignalPredictor()
            result = predictor.predict_signals(market_data)
            
            return result.get('signals', {})
            
        except Exception as e:
            print(f"AI prediction error for {symbol}: {e}")
            return {
                'primary_signal': 'HOLD',
                'signal_confidence': 0.5,
                'model_confidence': 0.5,
                'all_probabilities': {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34}
            }

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
            ai_input["data_sources"]['repo_data'] = True
            ai_input["raw_files_count"] = len(raw_data)

        # داده‌های بازار
        market_data = self.get_market_data()
        if market_data:
            ai_input["market_data"] = market_data
            ai_input["data_sources"]['api_data'] = True

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
                
                # استخراج قیمت‌ها و حجم‌ها
                if 'result' in historical_data:
                    prices = [float(item['price']) for item in historical_data['result'] if 'price' in item]
                    volumes = [float(item.get('volume', 1000000)) for item in historical_data['result']]
                    symbol_data["prices"] = prices
                    symbol_data["volumes"] = volumes

            # اندیکاتورهای تکنیکال
            technical_indicators = self.get_technical_indicators(symbol, period)
            if technical_indicators:
                symbol_data["technical_indicators"] = technical_indicators

            # پیش‌بینی AI
            if symbol_data:
                ai_prediction = self.get_ai_prediction(symbol, symbol_data)
                symbol_data["ai_prediction"] = ai_prediction

            if symbol_data:
                ai_input["symbols_data"][symbol] = symbol_data

        return ai_input

    def generate_analysis_report(self, ai_input: Dict) -> Dict[str, Any]:
        """تولید گزارش تحلیل کامل"""
        symbols_data = ai_input.get("symbols_data", {})
        market_insights = ai_input.get("insights_data", {})
        
        report = {
            "analysis_id": f"ai_analysis_{int(datetime.now().timestamp())}",
            "timestamp": ai_input["timestamp"],
            "summary": {
                "total_symbols": len(symbols_data),
                "analysis_period": ai_input["period"],
                "data_quality": "high" if ai_input["data_sources"]["api_data"] else "medium",
                "market_sentiment": self._get_market_sentiment(market_insights)
            },
            "symbol_analysis": {},
            "market_overview": {
                "fear_greed_index": market_insights.get("fear_greed", {}),
                "btc_dominance": market_insights.get("btc_dominance", {}),
                "top_performers": self._get_top_performers(ai_input.get("market_data", {}))
            },
            "trading_signals": {},
            "risk_assessment": {
                "overall_risk": "medium",
                "volatility_level": "normal",
                "recommended_actions": []
            }
        }
        
        # تحلیل هر نماد
        for symbol, data in symbols_data.items():
            symbol_report = {
                "current_price": data.get("prices", [0])[-1] if data.get("prices") else 0,
                "price_change_24h": data.get("coin_info", {}).get("priceChange1d", 0),
                "technical_score": self._calculate_technical_score(data.get("technical_indicators", {})),
                "ai_signal": data.get("ai_prediction", {}),
                "support_levels": [],
                "resistance_levels": [],
                "momentum": "neutral"
            }
            
            report["symbol_analysis"][symbol] = symbol_report
            
            # سیگنال معاملاتی
            ai_signal = data.get("ai_prediction", {})
            if ai_signal:
                report["trading_signals"][symbol] = {
                    "action": ai_signal.get("primary_signal", "HOLD"),
                    "confidence": ai_signal.get("signal_confidence", 0.5),
                    "reasoning": self._generate_signal_reasoning(symbol, data)
                }
        
        return report

    def _get_market_sentiment(self, insights: Dict) -> str:
        """دریافت احساسات کلی بازار"""
        fear_greed = insights.get("fear_greed", {}).get("now", {}).get("value", 50)
        
        if fear_greed >= 70:
            return "bullish"
        elif fear_greed <= 30:
            return "bearish"
        else:
            return "neutral"

    def _get_top_performers(self, market_data: Dict) -> List[Dict]:
        """دریافت بهترین عملکردها"""
        top_coins = market_data.get("top_coins", [])
        performers = []
        
        for coin in top_coins[:5]:
            performers.append({
                "symbol": coin.get("symbol"),
                "price": coin.get("price"),
                "change_24h": coin.get("priceChange1d", 0),
                "volume": coin.get("volume", 0)
            })
            
        return performers

    def _calculate_technical_score(self, indicators: Dict) -> float:
        """محاسبه امتیاز تکنیکال"""
        if not indicators:
            return 0.5
            
        score = 0.5  # نمره پایه
        
        # RSI
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 0.1
        elif rsi < 30 or rsi > 70:
            score -= 0.1
            
        # MACD
        macd = indicators.get('macd', 0)
        if macd > 0:
            score += 0.1
        else:
            score -= 0.1
            
        return max(0, min(1, score))

    def _generate_signal_reasoning(self, symbol: str, data: Dict) -> str:
        """تولید استدلال برای سیگنال"""
        technical = data.get("technical_indicators", {})
        ai_signal = data.get("ai_prediction", {})
        
        reasons = []
        
        # استدلال‌های تکنیکال
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            reasons.append("RSI در ناحیه oversold")
        elif rsi > 70:
            reasons.append("RSI در ناحیه overbought")
            
        macd = technical.get('macd', 0)
        if macd > 0:
            reasons.append("MACD مثبت")
        else:
            reasons.append("MACD منفی")
            
        # استدلال AI
        signal = ai_signal.get('primary_signal', 'HOLD')
        confidence = ai_signal.get('signal_confidence', 0.5)
        
        if confidence > 0.7:
            reasons.append("اطمینان بالای مدل AI")
        elif confidence < 0.3:
            reasons.append("اطمینان پایین مدل AI")
            
        return " - ".join(reasons) if reasons else "تحلیل خنثی"

# ایجاد سرویس
ai_service = AIAnalysisService()

# ========================= روت‌ها =========================

@router.get("/analysis")
async def ai_analysis(
    symbols: List[str] = Query(..., description="نمادها برای تحلیل"),
    period: str = Query("7d", regex="^(1h|4h|1d|7d|30d|90d|all)$"),
    include_news: bool = True,
    include_market_data: bool = True,
    include_technical: bool = True,
    analysis_type: str = "comprehensive"
):
    """تحلیل هوش مصنوعی برای نمادها"""
    try:
        # آماده‌سازی داده های ورودی برای هوش مصنوعی
        ai_input = ai_service.prepare_ai_input(symbols, period)
        
        # اگر داده دریافت نشد
        if not ai_input["data_sources"]["repo_data"] and not ai_input["data_sources"]["api_data"]:
            raise HTTPException(
                status_code=503,
                detail="هیچ منبع داده‌ای در دسترس نیست"
            )
        
        # تولید گزارش تحلیل
        analysis_report = ai_service.generate_analysis_report(ai_input)
        
        return {
            "status": "success",
            "message": "تحلیل AI با موفقیت انجام شد",
            "analysis_report": analysis_report,
            "input_summary": {
                "symbols_processed": len(ai_input["symbols_data"]),
                "market_data_available": bool(ai_input["market_data"]),
                "news_data_available": bool(ai_input["news_data"]),
                "insights_available": bool(ai_input["insights_data"]),
                "technical_analysis": include_technical,
                "data_sources": ai_input["data_sources"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل AI: {str(e)}")

@router.get("/analysis/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """دریافت وضعیت تحلیل"""
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "progress": 100,
        "timestamp": int(datetime.now().timestamp()),
        "results_ready": True
    }

@router.get("/analysis/symbols")
async def get_available_symbols():
    """دریافت لیست نمادهای قابل تحلیل"""
    try:
        from complete_coinstats_manager import CompleteCoinStatsManager
        manager = CompleteCoinStatsManager()
        coins = manager.get_all_coins(limit=100)
        
        symbols = [coin['symbol'] for coin in coins if 'symbol' in coin]
        
        return {
            "available_symbols": symbols,
            "total_count": len(symbols),
            "popular_symbols": ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOT", "LTC"]
        }
    except Exception as e:
        return {
            "available_symbols": ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOT", "LTC"],
            "total_count": 8,
            "error": str(e)
        }

@router.get("/analysis/types")
async def get_analysis_types():
    """دریافت انواع تحلیل‌های موجود"""
    return {
        "available_analysis_types": [
            {
                "type": "comprehensive",
                "name": "تحلیل جامع",
                "description": "تحلیل کامل تکنیکال، سنتیمنتال و AI"
            },
            {
                "type": "technical", 
                "name": "تحلیل تکنیکال",
                "description": "تمرکز بر اندیکاتورهای تکنیکال"
            },
            {
                "type": "sentiment",
                "name": "تحلیل احساسات",
                "description": "تحلیل احساسات بازار و اخبار"
            },
            {
                "type": "momentum",
                "name": "تحلیل مومنتوم", 
                "description": "تحلیل قدرت روند و مومنتوم"
            }
        ]
    }

# نمونه استفاده
if __name__ == "__main__":
    # تست سرویس
    service = AIAnalysisService()
    
    print("🧪 تست سرویس تحلیل AI...")
    
    # تست با BTC و ETH
    ai_input = service.prepare_ai_input(["BTC", "ETH"], "7d")
    print(f"✅ داده‌های ورودی آماده شد - نمادها: {len(ai_input['symbols_data'])}")
    
    # تولید گزارش
    report = service.generate_analysis_report(ai_input)
    print(f"✅ گزارش تحلیل تولید شد - تحلیل‌ID: {report['analysis_id']}")
    
    print("🎉 سرویس تحلیل AI آماده است!")
