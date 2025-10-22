# 📁 src/api/routes/trading.py

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from datetime import datetime
import asyncio

from ...main import CryptoAnalysisEngine
from ...core.technical_analysis.signal_engine import IntelligentSignalEngine
from ...core.risk_management.position_sizing import DynamicPositionSizing
from ..middleware import verify_api_key

router = APIRouter()

@router.get("/signals")
async def get_trading_signals(
    symbols: Optional[List[str]] = Query(None),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0),
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """دریافت سیگنال‌های معاملاتی"""
    try:
        if symbols is None:
            symbols = engine.config['default_symbols']
        
        results = await engine.run_analysis_pipeline(symbols=symbols)
        
        signals = []
        for symbol, analysis in results["results"].items():
            if "signals" in analysis:
                for signal_data in analysis["signals"]:
                    if signal_data["confidence"] >= min_confidence:
                        signals.append({
                            "symbol": symbol,
                            "signal_type": signal_data["signal_type"],
                            "confidence": signal_data["confidence"],
                            "price": signal_data["price"],
                            "targets": signal_data["targets"],
                            "stop_loss": signal_data["stop_loss"],
                            "risk_reward": signal_data["risk_reward_ratio"],
                            "time_horizon": signal_data["time_horizon"],
                            "reasons": signal_data["reasons"],
                            "timestamp": signal_data["timestamp"]
                        })
        
        # مرتب‌سازی بر اساس اعتماد
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "total_signals": len(signals),
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {str(e)}")

@router.get("/risk-assessment/{symbol}")
async def risk_assessment(
    symbol: str,
    position_size: float = Query(1000.0, gt=0),
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """ارزیابی ریسک برای یک نماد"""
    try:
        results = await engine.run_analysis_pipeline(symbols=[symbol])
        symbol_analysis = results["results"].get(symbol, {})
        
        if "signals" not in symbol_analysis or not symbol_analysis["signals"]:
            raise HTTPException(status_code=404, detail=f"No signals found for {symbol}")
        
        latest_signal = symbol_analysis["signals"][0]
        
        # شبیه‌سازی پورتفو
        mock_portfolio = {symbol: {"risk_amount": position_size * 0.1}}
        
        # محاسبه سایز پوزیشن
        market_data = {symbol: engine._prepare_multi_timeframe_data(symbol)["1h"]}
        position = engine.risk_manager.calculate_position_size(
            latest_signal, market_data, mock_portfolio
        )
        
        # شبیه‌سازی مونت کارلو
        from ...backtesting.monte_carlo import MonteCarloSimulator
        mc_simulator = MonteCarloSimulator()
        
        # ایجاد نتایج بک‌تست شبیه‌سازی شده
        class MockBacktestResult:
            def __init__(self):
                self.trades = [latest_signal]
        
        mc_result = mc_simulator.run_simulation(MockBacktestResult(), position_size)
        
        return {
            "symbol": symbol,
            "position_size": position.__dict__,
            "risk_metrics": {
                "value_at_risk": mc_result.value_at_risk,
                "expected_shortfall": mc_result.expected_shortfall,
                "probability_of_ruin": mc_result.probability_of_ruin,
                "max_drawdown": mc_result.worst_case / position_size * 100
            },
            "recommendation": "BUY" if latest_signal["signal_type"] in ["BUY", "STRONG_BUY"] else "SELL",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.post("/backtest")
async def run_backtest(
    strategy_config: Dict,
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """اجرای بک‌تست برای استراتژی"""
    try:
        results = await engine.run_backtest(strategy_config)
        
        return {
            "backtest_id": f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "strategy_config": strategy_config,
            "results": results["backtest_results"],
            "dashboard": results["dashboard"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@router.get("/portfolio-analysis")
async def portfolio_analysis(
    portfolio: Dict[str, float] = Query(...),
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """آنالیز پورتفوی معاملاتی"""
    try:
        total_value = sum(portfolio.values())
        symbols = list(portfolio.keys())
        
        # تحلیل هر نماد در پورتفو
        analysis_results = await engine.run_analysis_pipeline(symbols=symbols)
        
        portfolio_metrics = {
            "total_value": total_value,
            "diversification_score": len(symbols) / 10.0,  # نمره ساده
            "risk_assessment": {},
            "performance": {},
            "recommendations": []
        }
        
        for symbol, value in portfolio.items():
            symbol_analysis = analysis_results["results"].get(symbol, {})
            allocation = (value / total_value) * 100
            
            if "market_regime" in symbol_analysis:
                regime = symbol_analysis["market_regime"]["regime"]
                portfolio_metrics["risk_assessment"][symbol] = {
                    "allocation": allocation,
                    "regime": regime,
                    "confidence": symbol_analysis["market_regime"]["confidence"]
                }
            
            if "signals" in symbol_analysis and symbol_analysis["signals"]:
                latest_signal = symbol_analysis["signals"][0]
                portfolio_metrics["recommendations"].append({
                    "symbol": symbol,
                    "action": "HOLD" if latest_signal["signal_type"] == "HOLD" else "ADJUST",
                    "current_allocation": allocation,
                    "recommended_action": latest_signal["signal_type"],
                    "confidence": latest_signal["confidence"]
                })
        
        return {
            "portfolio": portfolio_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")
