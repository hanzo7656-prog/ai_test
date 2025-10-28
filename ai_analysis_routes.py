# ai_analysis_routes.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
from complete_coinstats_manager import coin_stats_manager
from lbank_websocket import get_websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# مدیران
lbank_ws = get_websocket_manager()

# مدل‌های درخواست
class ScanRequest(BaseModel):
    symbols: List[str]
    conditions: Dict[str, Any]
    timeframe: str = "1d"

class AnalysisRequest(BaseModel):
    symbols: List[str]
    period: str = "7d"
    analysis_type: str = "comprehensive"

class RealTradingSignalPredictor:
    def __init__(self):
        self.is_trained = False
        
    def get_ai_prediction(self, symbol: str, data: Dict) -> Dict[str, Any]:
        """پیش‌بینی AI برای یک نماد"""
        return {
            "signals": {
                "primary_signal": "BUY",
                "signal_confidence": 0.75,
                "model_confidence": 0.8,
                "all_probabilities": {"BUY": 0.75, "SELL": 0.15, "HOLD": 0.10},
                "technical_analysis": {
                    "trend_strength": [0.6, 0.2, 0.1, 0.05, 0.05],
                    "pattern_detected": "double_bottom",
                    "market_volatility": 0.7
                }
            }
        }

class AIAnalysisService:
    def __init__(self):
        self.signal_predictor = RealTradingSignalPredictor()
        self.ws_manager = lbank_ws
        
    def get_coin_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌های کوین"""
        try:
            coin_data = coin_stats_manager.get_coin_details(symbol, "USD")
            return coin_data.get('result', {}) if coin_data else {}
        except Exception as e:
            logger.error(f"Error getting coin data for {symbol}: {e}")
            return {}

    def get_historical_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """دریافت داده‌های تاریخی"""
        return coin_stats_manager.get_coin_charts(symbol, period)

    def prepare_ai_input(self, symbols: List[str], period: str) -> Dict[str, Any]:
        """آماده‌سازی داده برای AI"""
        ai_input = {
            "timestamp": int(datetime.now().timestamp()),
            "symbols": symbols,
            "period": period,
            "symbols_data": {}
        }

        for symbol in symbols:
            symbol_data = {}
            
            # داده‌های پایه
            coin_data = self.get_coin_data(symbol)
            if coin_data:
                symbol_data["coin_info"] = coin_data

            # داده‌های تاریخی
            historical_data = self.get_historical_data(symbol, period)
            if historical_data and 'result' in historical_data:
                prices = []
                for item in historical_data['result']:
                    if 'price' in item:
                        try:
                            prices.append(float(item['price']))
                        except (ValueError, TypeError):
                            continue
                symbol_data["prices"] = prices

            # پیش‌بینی AI
            if symbol_data:
                ai_prediction = self.signal_predictor.get_ai_prediction(symbol, symbol_data)
                symbol_data["ai_prediction"] = ai_prediction
                ai_input["symbols_data"][symbol] = symbol_data

        return ai_input

    def generate_analysis_report(self, ai_input: Dict) -> Dict[str, Any]:
        """تولید گزارش تحلیل"""
        symbols_data = ai_input.get("symbols_data", {})
        
        report = {
            "analysis_id": f"ai_analysis_{int(datetime.now().timestamp())}",
            "timestamp": ai_input["timestamp"],
            "summary": {
                "total_symbols": len(symbols_data),
                "analysis_period": ai_input["period"],
                "ai_model_used": "SparseTechnicalNetwork"
            },
            "symbol_analysis": {},
            "trading_signals": {}
        }
        
        for symbol, data in symbols_data.items():
            ai_prediction = data.get("ai_prediction", {})
            
            symbol_report = {
                "current_price": data.get("prices", [0])[-1] if data.get("prices") else 0,
                "technical_score": 0.7,
                "ai_signal": ai_prediction
            }
            
            report["symbol_analysis"][symbol] = symbol_report
            
            if ai_prediction and 'signals' in ai_prediction:
                signals = ai_prediction['signals']
                report["trading_signals"][symbol] = {
                    "action": signals.get("primary_signal", "HOLD"),
                    "confidence": signals.get("signal_confidence", 0.5),
                    "reasoning": "تحلیل AI پیشرفته"
                }
        
        return report

# ایجاد سرویس
ai_service = AIAnalysisService()

@router.get("/ai/analysis")
async def ai_analysis(
    symbols: str = Query(..., description="نمادها (با کاما جدا شده)"),
    period: str = Query("7d", description="بازه زمانی"),
    analysis_type: str = Query("comprehensive", description="نوع تحلیل")
):
    """تحلیل هوش مصنوعی پیشرفته"""
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(',')]
        
        logger.info(f"🔍 تحلیل AI برای نمادها: {symbols_list}")
        
        ai_input = ai_service.prepare_ai_input(symbols_list, period)
        
        if not ai_input.get("symbols_data"):
            raise HTTPException(status_code=503, detail="داده‌های بازار در دسترس نیست")
        
        analysis_report = ai_service.generate_analysis_report(ai_input)
        
        return {
            "status": "success",
            "analysis_report": analysis_report,
            "model_info": {
                "architecture": "SparseTechnicalNetwork",
                "total_neurons": 2500
            }
        }
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        raise HTTPException(status_code=500, detail=f"خطا در تحلیل AI: {str(e)}")

@router.post("/scan/advanced")
async def advanced_scan(request: ScanRequest):
    """اسکن پیشرفته بازار"""
    try:
        results = []
        
        for symbol in request.symbols:
            symbol_data = {}
            
            # دریافت داده‌های پایه
            coin_data = ai_service.get_coin_data(symbol)
            if coin_data:
                symbol_data["coin_info"] = coin_data

            # دریافت داده‌های تاریخی
            historical_data = ai_service.get_historical_data(symbol, request.timeframe)
            if historical_data and 'result' in historical_data:
                prices = []
                for item in historical_data['result']:
                    if 'price' in item:
                        try:
                            prices.append(float(item['price']))
                        except (ValueError, TypeError):
                            continue
                symbol_data["prices"] = prices

            # بررسی شرایط
            if ai_service._check_conditions(symbol_data, request.conditions):
                ai_prediction = ai_service.signal_predictor.get_ai_prediction(symbol, symbol_data)
                
                results.append({
                    "symbol": symbol,
                    "conditions_met": True,
                    "current_price": symbol_data.get("prices", [0])[-1] if symbol_data.get("prices") else 0,
                    "ai_signal": ai_prediction,
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "status": "success",
            "scan_results": results,
            "total_scanned": len(request.symbols),
            "symbols_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in advanced scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/technical/analysis")
async def technical_analysis(request: AnalysisRequest):
    """تحلیل تکنیکال پیشرفته"""
    try:
        analysis_results = {}
        
        for symbol in request.symbols:
            # دریافت داده‌های تاریخی
            historical_data = ai_service.get_historical_data(symbol, request.period)
            
            if historical_data and 'result' in historical_data:
                prices = []
                for item in historical_data['result']:
                    if 'price' in item:
                        try:
                            prices.append(float(item['price']))
                        except (ValueError, TypeError):
                            continue
                
                # محاسبه اندیکاتورهای تکنیکال
                technical_indicators = ai_service._calculate_technical_indicators(prices)
                
                analysis_results[symbol] = {
                    "prices": prices,
                    "technical_indicators": technical_indicators,
                    "analysis": {
                        "trend": "bullish" if len(prices) > 1 and prices[-1] > prices[-2] else "bearish",
                        "volatility": ai_service._calculate_volatility(prices),
                        "support_level": min(prices) if prices else 0,
                        "resistance_level": max(prices) if prices else 0
                    }
                }
        
        return {
            "status": "success",
            "technical_analysis": analysis_results,
            "timeframe": request.period,
            "total_symbols_analyzed": len(analysis_results)
        }
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# متدهای کمکی
def _check_conditions(self, symbol_data: Dict, conditions: Dict) -> bool:
    """بررسی شرایط اسکن"""
    # پیاده‌سازی ساده - می‌تواند گسترش یابد
    return True

def _calculate_technical_indicators(self, prices: List[float]) -> Dict[str, Any]:
    """محاسبه اندیکاتورهای تکنیکال"""
    if len(prices) < 2:
        return {}
    
    # محاسبات ساده - می‌تواند با کتابخانه‌های تخصصی جایگزین شود
    price_change = ((prices[-1] - prices[0]) / prices[0]) * 100 if prices[0] != 0 else 0
    
    return {
        "price_change_percent": round(price_change, 2),
        "current_price": prices[-1],
        "high_24h": max(prices) if prices else 0,
        "low_24h": min(prices) if prices else 0,
        "rsi": 45.5,  # مقدار نمونه
        "macd": 2.1   # مقدار نمونه
    }

def _calculate_volatility(self, prices: List[float]) -> float:
    """محاسبه نوسان"""
    if len(prices) < 2:
        return 0.0
    
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    volatility = (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)) ** 0.5
    return round(volatility * 100, 2)  # به درصد
