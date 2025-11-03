# main.py - با اتصال واقعی به CoinStats API
from fastapi import FastAPI, HTTPException, APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import time
import psutil
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CryptoAI API", version="3.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ایجاد پوشه frontend
os.makedirs("frontend", exist_ok=True)

# مدل‌های درخواست
class ScanRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1h"
    scan_mode: str = "ai"
    conditions: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    symbols: List[str]
    period: str = "7d"
    analysis_type: str = "comprehensive"
    indicators: Optional[List[str]] = None

class TechnicalAnalysisRequest(BaseModel):
    symbols: List[str]
    period: str = "7d"
    analysis_type: str = "comprehensive"

class AITrainingRequest(BaseModel):
    symbols: List[str]
    epochs: int = 30
    training_type: str = "technical"

# ==================== ایمپورت مدیر CoinStats ====================

try:
    from complete_coinstats_manager import coin_stats_manager
    COINSTATS_AVAILABLE = True
    logger.info("✅ CoinStats Manager loaded successfully")
except ImportError as e:
    COINSTATS_AVAILABLE = False
    logger.error(f"❌ CoinStats Manager import failed: {e}")

# ==================== سیستم تحلیل هوش مصنوعی با داده واقعی ====================

class RealAIAnalyzer:
    """سیستم تحلیل هوش مصنوعی با داده‌های واقعی از CoinStats"""
    
    def __init__(self):
        self.analysis_count = 0
        self.coin_stats = coin_stats_manager if COINSTATS_AVAILABLE else None
        
    async def analyze_symbol(self, symbol: str, period: str = "1w") -> Dict[str, Any]:
        """تحلیل نماد با داده‌های واقعی"""
        self.analysis_count += 1
        
        try:
            # دریافت داده‌های واقعی از CoinStats
            coin_details = self.coin_stats.get_coin_details(symbol, "USD")
            coin_charts = self.coin_stats.get_coin_charts(symbol, period)
            
            # تحلیل داده‌های واقعی
            analysis_result = self._analyze_real_data(coin_details, coin_charts, symbol, period)
            return analysis_result
            
        except Exception as e:
            logger.error(f"خطا در تحلیل {symbol}: {str(e)}")
            return self._get_fallback_analysis(symbol)
    
    def _analyze_real_data(self, coin_details: Dict, coin_charts: Dict, symbol: str, period: str) -> Dict[str, Any]:
        """تحلیل داده‌های واقعی"""
        # استخراج اطلاعات از داده‌های واقعی
        price_info = self._extract_price_info(coin_details)
        technical_data = self._extract_technical_data(coin_charts, coin_details)
        market_data = self._extract_market_data(coin_details)
        
        # تولید سیگنال بر اساس داده‌های واقعی
        signal = self._generate_real_signal(price_info, technical_data, market_data)
        
        return {
            "symbol": symbol,
            "real_data": True,
            "source": "coinstats",
            "timestamp": datetime.now().isoformat(),
            "price_info": price_info,
            "technical_analysis": technical_data,
            "market_data": market_data,
            "trading_signal": signal,
            "risk_assessment": self._assess_risk(price_info, technical_data),
            "ai_insights": self._generate_ai_insights(price_info, technical_data, market_data)
        }
    
    def _extract_price_info(self, coin_details: Dict) -> Dict[str, Any]:
        """استخراج اطلاعات قیمت از داده‌های واقعی"""
        try:
            result = coin_details.get('result', {})
            return {
                "current_price": result.get('price', 0),
                "price_change_24h": result.get('priceChange1d', 0),
                "price_change_percent_24h": result.get('priceChange1d', 0),
                "high_24h": result.get('high', 0),
                "low_24h": result.get('low', 0),
                "volume_24h": result.get('volume', 0),
                "market_cap": result.get('marketCap', 0),
                "rank": result.get('rank', 0)
            }
        except Exception as e:
            logger.error(f"خطا در استخراج اطلاعات قیمت: {e}")
            return {
                "current_price": 0,
                "price_change_24h": 0,
                "price_change_percent_24h": 0,
                "high_24h": 0,
                "low_24h": 0,
                "volume_24h": 0,
                "market_cap": 0,
                "rank": 0
            }
    
    def _extract_technical_data(self, coin_charts: Dict, coin_details: Dict) -> Dict[str, Any]:
        """استخراج داده‌های تکنیکال"""
        try:
            # تحلیل داده‌های چارت
            chart_data = coin_charts.get('result', [])
            prices = [point.get('price', 0) for point in chart_data if point.get('price')]
            
            if prices:
                current_price = prices[-1]
                min_price = min(prices)
                max_price = max(prices)
                
                # محاسبه اندیکاتورها
                rsi = self._calculate_rsi(prices)
                trend = self._analyze_trend(prices)
                
                return {
                    "rsi": rsi,
                    "trend": trend,
                    "support_level": min_price * 0.95,
                    "resistance_level": max_price * 1.05,
                    "volatility": self._calculate_volatility(prices),
                    "momentum": "صعودی" if trend == "up" else "نزولی" if trend == "down" else "خنثی",
                    "data_points": len(prices)
                }
            else:
                return {
                    "rsi": 50,
                    "trend": "unknown",
                    "support_level": 0,
                    "resistance_level": 0,
                    "volatility": 0,
                    "momentum": "نامشخص",
                    "data_points": 0
                }
                
        except Exception as e:
            logger.error(f"خطا در استخراج داده‌های تکنیکال: {e}")
            return {
                "rsi": 50,
                "trend": "unknown",
                "support_level": 0,
                "resistance_level": 0,
                "volatility": 0,
                "momentum": "نامشخص",
                "data_points": 0
            }
    
    def _extract_market_data(self, coin_details: Dict) -> Dict[str, Any]:
        """استخراج داده‌های بازار"""
        try:
            result = coin_details.get('result', {})
            return {
                "total_supply": result.get('totalSupply', 0),
                "available_supply": result.get('availableSupply', 0),
                "website": result.get('websiteUrl', ''),
                "explorers": result.get('explorers', []),
                "social_media": {
                    "twitter": result.get('twitterUrl', ''),
                    "reddit": result.get('redditUrl', '')
                }
            }
        except Exception as e:
            logger.error(f"خطا در استخراج داده‌های بازار: {e}")
            return {
                "total_supply": 0,
                "available_supply": 0,
                "website": "",
                "explorers": [],
                "social_media": {}
            }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """محاسبه RSI از داده‌های واقعی"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def _analyze_trend(self, prices: List[float]) -> str:
        """تحلیل روند از داده‌های واقعی"""
        if len(prices) < 5:
            return "unknown"
        
        recent_prices = prices[-5:]
        if recent_prices[-1] > recent_prices[0]:
            return "up"
        elif recent_prices[-1] < recent_prices[0]:
            return "down"
        else:
            return "sideways"
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """محاسبه نوسان"""
        if len(prices) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.0
        
        volatility = (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)) ** 0.5
        return round(volatility * 100, 2)
    
    def _generate_real_signal(self, price_info: Dict, technical_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """تولید سیگنال واقعی"""
        rsi = technical_data.get('rsi', 50)
        trend = technical_data.get('trend', 'unknown')
        price_change = price_info.get('price_change_percent_24h', 0)
        
        # منطق سیگنال‌دهی پیشرفته
        if rsi < 30 and trend == "up" and price_change > -5:
            signal = "BUY"
            confidence = 0.85
            reasoning = "اشباع فروش با روند صعودی و ثبات قیمتی"
        elif rsi > 70 and trend == "down" and price_change < 5:
            signal = "SELL"
            confidence = 0.75
            reasoning = "اشباع خرید با روند نزولی"
        elif 40 < rsi < 60 and abs(price_change) < 3:
            signal = "HOLD"
            confidence = 0.70
            reasoning = "بازار در حالت تعادل و ثبات"
        else:
            signal = "HOLD"
            confidence = 0.60
            reasoning = "سیگنال واضحی وجود ندارد - نیاز به تحلیل بیشتر"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "risk_level": "low" if confidence > 0.8 else "medium" if confidence > 0.6 else "high",
            "timeframe": "کوتاه مدت" if technical_data.get('volatility', 0) > 10 else "میان مدت"
        }
    
    def _assess_risk(self, price_info: Dict, technical_data: Dict) -> Dict[str, Any]:
        """ارزیابی ریسک"""
        volatility = technical_data.get('volatility', 0)
        volume = price_info.get('volume_24h', 0)
        market_cap = price_info.get('market_cap', 0)
        
        risk_score = (volatility * 0.4) + (max(0, 10 - (volume / max(market_cap, 1)) * 1000000) * 0.6)
        
        if risk_score < 3:
            level = "کم"
            color = "success"
        elif risk_score < 7:
            level = "متوسط"
            color = "warning"
        else:
            level = "زیاد"
            color = "danger"
        
        return {
            "risk_score": round(risk_score, 2),
            "risk_level": level,
            "color": color,
            "factors": [
                f"نوسان: {volatility}%",
                f"حجم معاملات: {volume:,.0f}",
                f"ارزش بازار: {market_cap:,.0f}"
            ]
        }
    
    def _generate_ai_insights(self, price_info: Dict, technical_data: Dict, market_data: Dict) -> List[str]:
        """تولید بینش‌های هوش مصنوعی"""
        insights = []
        
        # تحلیل بنیادی
        if price_info.get('market_cap', 0) > 1000000000:  # بیش از 1 میلیارد
            insights.append("💰 ارزش بازار بالا - پایداری بیشتر")
        
        if technical_data.get('rsi', 50) < 35:
            insights.append("📉 شرایط اشباع فروش - فرصت خرید")
        elif technical_data.get('rsi', 50) > 65:
            insights.append("📈 شرایط اشباع خرید - احتیاط لازم")
        
        if technical_data.get('volatility', 0) > 15:
            insights.append("⚡ نوسان بالا - ریسک بیشتر")
        
        if price_info.get('volume_24h', 0) > 100000000:  # حجم بالا
            insights.append("🔊 نقدشوندگی عالی")
        
        if len(insights) == 0:
            insights.append("📊 بازار در حالت عادی - نظارت ادامه دار")
        
        return insights
    
    def _get_fallback_analysis(self, symbol: str) -> Dict[str, Any]:
        """تحلیل جایگزین در صورت خطا"""
        return {
            "symbol": symbol,
            "real_data": False,
            "source": "fallback",
            "timestamp": datetime.now().isoformat(),
            "price_info": {
                "current_price": 0,
                "price_change_24h": 0,
                "high_24h": 0,
                "low_24h": 0,
                "volume_24h": 0,
                "market_cap": 0
            },
            "technical_analysis": {
                "rsi": 50,
                "trend": "unknown",
                "support_level": 0,
                "resistance_level": 0,
                "volatility": 0
            },
            "trading_signal": {
                "signal": "HOLD",
                "confidence": 0.5,
                "reasoning": "داده در دسترس نیست",
                "risk_level": "high"
            },
            "error": "عدم دسترسی به داده‌های واقعی"
        }

# ==================== سیستم مدیریت بازار ====================

class MarketManager:
    """مدیریت داده‌های بازار واقعی"""
    
    def __init__(self):
        self.coin_stats = coin_stats_manager if COINSTATS_AVAILABLE else None
        self.ai_analyzer = RealAIAnalyzer()
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """دریافت نمای کلی بازار"""
        try:
            # دریافت داده‌های مختلف از CoinStats
            coins_list = self.coin_stats.get_coins_list(limit=50)
            fear_greed = self.coin_stats.get_fear_greed()
            btc_dominance = self.coin_stats.get_btc_dominance()
            
            return {
                "status": "success",
                "real_data": True,
                "timestamp": datetime.now().isoformat(),
                "market_summary": {
                    "total_coins": len(coins_list.get('result', [])),
                    "fear_greed_index": fear_greed.get('result', {}),
                    "btc_dominance": btc_dominance.get('result', {}),
                    "market_trend": self._analyze_market_trend(coins_list.get('result', []))
                },
                "top_performers": self._get_top_performers(coins_list.get('result', [])),
                "market_health": self._assess_market_health(coins_list.get('result', []))
            }
        except Exception as e:
            logger.error(f"خطا در دریافت نمای بازار: {e}")
            return {
                "status": "error",
                "real_data": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_market_trend(self, coins: List[Dict]) -> str:
        """تحلیل روند کلی بازار"""
        if not coins:
            return "unknown"
        
        positive_changes = sum(1 for coin in coins if coin.get('priceChange1d', 0) > 0)
        total_coins = len(coins)
        
        if positive_changes / total_coins > 0.7:
            return "bullish"
        elif positive_changes / total_coins < 0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _get_top_performers(self, coins: List[Dict], count: int = 5) -> List[Dict]:
        """دریافت بهترین عملکردها"""
        if not coins:
            return []
        
        sorted_coins = sorted(coins, key=lambda x: x.get('priceChange1d', 0), reverse=True)
        return [
            {
                "symbol": coin.get('id', '').upper(),
                "price": coin.get('price', 0),
                "change_24h": coin.get('priceChange1d', 0),
                "volume": coin.get('volume', 0)
            }
            for coin in sorted_coins[:count]
        ]
    
    def _assess_market_health(self, coins: List[Dict]) -> Dict[str, Any]:
        """ارزیابی سلامت بازار"""
        if not coins:
            return {"score": 0, "status": "unknown"}
        
        total_volume = sum(coin.get('volume', 0) for coin in coins)
        avg_volume = total_volume / len(coins)
        
        # محاسبه نمره سلامت
        volume_score = min(100, (avg_volume / 1000000) * 10)  # نرمال‌سازی حجم
        diversity_score = min(100, len(coins) * 2)  # نمره تنوع
        
        health_score = (volume_score + diversity_score) / 2
        
        if health_score > 80:
            status = "عالی"
        elif health_score > 60:
            status = "خوب"
        elif health_score > 40:
            status = "متوسط"
        else:
            status = "ضعیف"
        
        return {
            "score": round(health_score, 2),
            "status": status,
            "factors": [
                f"حجم معاملات: {volume_score:.1f}%",
                f"تنوع بازار: {diversity_score:.1f}%"
            ]
        }

# ==================== ایجاد نمونه‌ها ====================

market_manager = MarketManager()
real_ai_analyzer = RealAIAnalyzer()

# ==================== روت‌های اصلی ====================

system_router = APIRouter(prefix="/api/system", tags=["system"])
ai_router = APIRouter(prefix="/api/ai", tags=["ai"])
market_router = APIRouter(prefix="/api/market", tags=["market"])

# روت‌های سیستم
@system_router.get("/health")
async def health_check():
    """سلامت سیستم"""
    return {
        "status": "healthy",
        "real_data": COINSTATS_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "coinstats_api": "active" if COINSTATS_AVAILABLE else "inactive",
            "ai_analyzer": "active",
            "market_manager": "active"
        }
    }

@system_router.get("/status")
async def system_status():
    """وضعیت سیستم"""
    return {
        "status": "running",
        "version": "3.0.0",
        "real_data": COINSTATS_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "analysis_count": real_ai_analyzer.analysis_count,
        "features": [
            "تحلیل واقعی بازار",
            "داده‌های زنده CoinStats",
            "سیگنال‌های هوش مصنوعی",
            "مدیریت ریسک پیشرفته"
        ]
    }

# روت‌های هوش مصنوعی
@ai_router.post("/scan")
async def ai_scan(request: ScanRequest):
    """اسکن هوشمند بازار با داده‌های واقعی"""
    try:
        results = []
        
        for symbol in request.symbols:
            analysis = await real_ai_analyzer.analyze_symbol(symbol, request.timeframe)
            results.append(analysis)
        
        return {
            "status": "success",
            "real_data": True,
            "scan_results": results,
            "total_scanned": len(results),
            "successful_scans": len([r for r in results if r.get('real_data', False)]),
            "scan_mode": request.scan_mode,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در اسکن AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در اسکن: {str(e)}")

@ai_router.get("/analysis")
async def ai_analysis(
    symbols: str = Query("BTC,ETH", description="نمادها با کاما جدا شده"),
    period: str = Query("1w", description="بازه زمانی"),
    analysis_type: str = Query("comprehensive", description="نوع تحلیل")
):
    """تحلیل پیشرفته AI"""
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
        
        analysis_results = {}
        for symbol in symbols_list:
            analysis = await real_ai_analyzer.analyze_symbol(symbol, period)
            analysis_results[symbol] = analysis
        
        return {
            "status": "success",
            "real_data": True,
            "analysis_report": {
                "analysis_id": f"ai_analysis_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_symbols": len(symbols_list),
                    "analysis_period": period,
                    "analysis_type": analysis_type,
                    "real_data_ratio": f"{len([a for a in analysis_results.values() if a.get('real_data', False)])}/{len(analysis_results)}"
                },
                "symbol_analysis": analysis_results,
                "market_context": await market_manager.get_market_overview()
            }
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل: {str(e)}")

@ai_router.get("/analysis/quick")
async def quick_analysis(
    symbols: str = Query("BTC,ETH,ADA", description="نمادها با کاما جدا شده"),
    period: str = Query("24h", description="بازه زمانی")
):
    """تحلیل سریع"""
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
        
        quick_results = []
        for symbol in symbols_list[:10]:  # حداکثر 10 نماد
            analysis = await real_ai_analyzer.analyze_symbol(symbol, period)
            
            quick_results.append({
                "symbol": symbol,
                "price": analysis["price_info"]["current_price"],
                "change_24h": analysis["price_info"]["price_change_24h"],
                "signal": analysis["trading_signal"]["signal"],
                "confidence": analysis["trading_signal"]["confidence"],
                "risk": analysis["trading_signal"]["risk_level"],
                "real_data": analysis["real_data"]
            })
        
        return {
            "status": "success",
            "real_data": True,
            "quick_analysis": quick_results,
            "timestamp": datetime.now().isoformat(),
            "period": period
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل سریع: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل سریع: {str(e)}")

@ai_router.post("/technical/analysis")
async def technical_analysis(request: TechnicalAnalysisRequest):
    """تحلیل تکنیکال پیشرفته"""
    try:
        technical_results = {}
        
        for symbol in request.symbols:
            analysis = await real_ai_analyzer.analyze_symbol(symbol, request.period)
            technical_results[symbol] = {
                "symbol": symbol,
                "technical_indicators": analysis["technical_analysis"],
                "price_action": analysis["price_info"],
                "signal_strength": analysis["trading_signal"]["confidence"],
                "trend_analysis": analysis["technical_analysis"]["trend"]
            }
        
        return {
            "status": "success",
            "real_data": True,
            "technical_analysis": technical_results,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": request.analysis_type
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل تکنیکال: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل تکنیکال: {str(e)}")

# روت‌های بازار
@market_router.get("/overview")
async def market_overview():
    """نمای کلی بازار"""
    return await market_manager.get_market_overview()

@market_router.get("/prices")
async def market_prices(
    symbols: str = Query("BTC,ETH,ADA,SOL,DOT", description="نمادها با کاما جدا شده")
):
    """قیمت‌های بازار"""
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
        
        prices = {}
        for symbol in symbols_list:
            analysis = await real_ai_analyzer.analyze_symbol(symbol, "24h")
            prices[symbol] = {
                "price": analysis["price_info"]["current_price"],
                "change_24h": analysis["price_info"]["price_change_24h"],
                "volume": analysis["price_info"]["volume_24h"],
                "real_data": analysis["real_data"],
                "timestamp": analysis["timestamp"]
            }
        
        return {
            "status": "success",
            "real_data": True,
            "prices": prices,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت قیمت‌ها: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت قیمت‌ها: {str(e)}")

@market_router.get("/fear-greed")
async def fear_greed_index():
    """شاخص ترس و طمع"""
    try:
        if COINSTATS_AVAILABLE:
            fear_greed = coin_stats_manager.get_fear_greed()
            return {
                "status": "success",
                "real_data": True,
                "fear_greed_index": fear_greed.get('result', {}),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "real_data": False,
                "error": "CoinStats API در دسترس نیست",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"خطا در دریافت شاخص ترس و طمع: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت شاخص: {str(e)}")

# ==================== روت‌های عمومی ====================

@app.get("/api/info")
async def system_info():
    """اطلاعات سیستم"""
    return {
        "name": "Crypto AI Trading System",
        "version": "3.0.0",
        "status": "running",
        "real_data": COINSTATS_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "description": "سیستم پیشرفته تحلیل بازار با داده‌های واقعی از CoinStats API",
        "features": [
            "اتصال مستقیم به CoinStats API",
            "تحلیل تکنیکال پیشرفته",
            "سیگنال‌های هوش مصنوعی مبتنی بر داده واقعی",
            "مدیریت ریسک هوشمند",
            "نمای کلی بازار زنده"
        ],
        "statistics": {
            "total_analyses": real_ai_analyzer.analysis_count,
            "real_data_available": COINSTATS_AVAILABLE,
            "active_services": ["AI Analyzer", "Market Manager", "CoinStats API"]
        }
    }

@app.get("/api/health")
async def root_health():
    """سلامت ریشه"""
    return await health_check()

# ثبت روت‌ها
app.include_router(system_router)
app.include_router(ai_router)
app.include_router(market_router)

# ==================== سرویس فرانت‌اند ====================

@app.get("/")
async def serve_frontend():
    """سرویس دهی فرانت‌اند"""
    try:
        return FileResponse("frontend/index.html")
    except Exception as e:
        logger.error(f"خطا در بارگذاری فرانت‌اند: {str(e)}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "فایل فرانت‌اند یافت نشد",
                "detail": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/{full_path:path}")
async def serve_all_routes(full_path: str):
    """مدیریت تمام مسیرها برای SPA"""
    if full_path.startswith('api/'):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Endpoint not found",
                "path": full_path,
                "available_endpoints": [
                    "/api/health",
                    "/api/system/status",
                    "/api/ai/scan",
                    "/api/ai/analysis", 
                    "/api/ai/technical/analysis",
                    "/api/ai/analysis/quick",
                    "/api/market/overview",
                    "/api/market/prices",
                    "/api/market/fear-greed",
                    "/api/info"
                ],
                "timestamp": datetime.now().isoformat()
            }
        )
    else:
        try:
            return FileResponse("frontend/index.html")
        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Page not found",
                    "path": full_path,
                    "detail": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
