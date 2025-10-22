# 📁 src/api/routes/analysis.py

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import asyncio

from ...main import CryptoAnalysisEngine
from ..middleware import verify_api_key

router = APIRouter()

@router.get("/symbols/{symbol}")
async def analyze_symbol(
    symbol: str,
    timeframes: Optional[List[str]] = Query(["1h", "4h", "1d"]),
    include_ai: bool = True,
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """آنالیز کامل یک نماد"""
    try:
        result = await engine.run_analysis_pipeline(symbols=[symbol])
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis": result["results"].get(symbol, {}),
            "summary": result["summary"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/market-overview")
async def market_overview(
    symbols: Optional[List[str]] = Query(None),
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """نمای کلی بازار"""
    try:
        if symbols is None:
            symbols = engine.config['default_symbols'][:5]  # 5 نماد اول
        
        results = await engine.run_analysis_pipeline(symbols=symbols)
        
        # خلاصه‌سازی نتایج
        overview = {
            "total_symbols": len(results["results"]),
            "buy_signals": 0,
            "sell_signals": 0,
            "strong_buy_signals": 0,
            "strong_sell_signals": 0,
            "market_regimes": {},
            "top_opportunities": []
        }
        
        for symbol, analysis in results["results"].items():
            if "signals" in analysis:
                for signal in analysis["signals"]:
                    if signal["signal_type"] in ["BUY", "STRONG_BUY"]:
                        overview["buy_signals"] += 1
                        if signal["signal_type"] == "STRONG_BUY":
                            overview["strong_buy_signals"] += 1
                            overview["top_opportunities"].append({
                                "symbol": symbol,
                                "signal": signal["signal_type"],
                                "confidence": signal["confidence"],
                                "price": signal["price"]
                            })
                    else:
                        overview["sell_signals"] += 1
                        if signal["signal_type"] == "STRONG_SELL":
                            overview["strong_sell_signals"] += 1
            
            if "market_regime" in analysis:
                regime = analysis["market_regime"]["regime"]
                overview["market_regimes"][regime] = overview["market_regimes"].get(regime, 0) + 1
        
        # مرتب‌سازی فرصت‌های برتر
        overview["top_opportunities"] = sorted(
            overview["top_opportunities"],
            key=lambda x: x["confidence"],
            reverse=True
        )[:3]  # 3 فرصت برتر
        
        return overview
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")

@router.get("/technical/{symbol}")
async def technical_analysis(
    symbol: str,
    indicators: Optional[List[str]] = Query(["RSI", "MACD", "BBANDS", "VOLUME"]),
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """تحلیل تکنیکال پیشرفته"""
    try:
        # دریافت داده‌های خام
        raw_data = engine.data_manager.get_coins_data(symbols=[symbol], limit=100)
        if not raw_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # پردازش داده
        processed_data = engine.processing_pipeline.process_raw_data(raw_data)
        
        # تحلیل چندزمانه
        multi_tf_data = engine._prepare_multi_timeframe_data(symbol)
        mt_analysis = engine.multi_timeframe_analyzer.analyze_symbol(symbol, multi_tf_data)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price_data": {
                "current": processed_data["result"][0]["price"] if processed_data["result"] else 0,
                "high": max([c["price_high"] for c in processed_data["result"]]) if processed_data["result"] else 0,
                "low": min([c["price_low"] for c in processed_data["result"]]) if processed_data["result"] else 0
            },
            "multi_timeframe_analysis": mt_analysis,
            "technical_indicators": engine._extract_technical_indicators(processed_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

@router.get("/ai-predictions/{symbol}")
async def ai_predictions(
    symbol: str,
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """پیش‌بینی‌های هوش مصنوعی"""
    try:
        raw_data = engine.data_manager.get_coins_data(symbols=[symbol], limit=100)
        if not raw_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        processed_data = engine.processing_pipeline.process_raw_data(raw_data)
        
        # پیش‌بینی رژیم بازار
        regime_prediction = engine.regime_classifier.predict_regime(processed_data)
        
        # پیش‌بینی الگو
        pattern_prediction = engine.pattern_predictor.predict_pattern(processed_data)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_regime": regime_prediction,
            "pattern_prediction": pattern_prediction,
            "ai_confidence": {
                "regime": regime_prediction.get("confidence", 0),
                "pattern": pattern_prediction.get("confidence", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI predictions failed: {str(e)}")
