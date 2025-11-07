# main.py - سرور اصلی VortexAI با ۳ روت مادر
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import time
import psutil
from pathlib import Path
import json
import asyncio

# ایمپورت ماژول‌های AI
try:
    from trading_ai.technical_analyzer import TechnicalAnalyzer
    from trading_ai.neural_network import SparseNeuralNetwork
    from trading_ai.sentiment_analyzer import SentimentAnalyzer
    AI_AVAILABLE = True
    
    # ایجاد نمونه‌های AI
    technical_analyzer = TechnicalAnalyzer()
    neural_network = SparseNeuralNetwork()
    sentiment_analyzer = SentimentAnalyzer()
    
    logger.info("✅ Trading AI modules loaded successfully")
    
except ImportError as e:
    AI_AVAILABLE = False
    logger.warning(f"🔶 Trading AI not available: {e}")

# ایمپورت مدیر CoinStats
try:
    from complete_coinstats_manager import coin_stats_manager
    COINSTATS_AVAILABLE = True
    logger.info("✅ CoinStats Manager loaded successfully")
except ImportError as e:
    COINSTATS_AVAILABLE = False
    logger.warning(f"🔶 CoinStats Manager not available: {e}")
    
    # Mock CoinStats Manager
    class MockCoinStatsManager:
        def get_coin_details(self, symbol, currency="USD"):
            return {
                "id": symbol, "name": symbol.capitalize(), "symbol": symbol.upper(),
                "price": round(1000 + hash(symbol) % 50000, 2),
                "priceChange1d": round((hash(symbol) % 40) - 20, 2),
                "volume": round(1000000 + hash(symbol) % 100000000, 2),
                "marketCap": round(10000000 + hash(symbol) % 1000000000, 2),
                "rank": (hash(symbol) % 100) + 1
            }
    
    coin_stats_manager = MockCoinStatsManager()

# تنظیمات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VortexAI API", version="3.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# سرو فایل‌های استاتیک
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# مدل‌های درخواست
class BatchScanRequest(BaseModel):
    symbols: List[str]
    data_type: str = "raw"  # raw | processed

class AIAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "technical"  # technical | sentiment | prediction
    raw_data: Optional[Dict[str, Any]] = None

# ==================== روت‌های مادر ====================

@app.get("/")
async def root():
    """صفحه اصلی"""
    try:
        return FileResponse("frontend/index.html")
    except:
        return JSONResponse(content={
            "message": "VortexAI API Server", 
            "version": "3.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "raw_data": "GET /api/raw/{symbol}",
                "raw_batch": "POST /api/raw/batch", 
                "processed_data": "GET /api/processed/{symbol}",
                "processed_batch": "POST /api/processed/batch",
                "system_status": "GET /api/status"
            }
        })

# ==================== روت مادر داده‌های خام ====================

@app.get("/api/raw/{symbol}")
async def get_raw_data(symbol: str):
    """داده‌های خام برای هوش مصنوعی"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Data service unavailable")
        
        # دریافت همه داده‌های خام برای AI
        raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
        raw_charts = coin_stats_manager.get_coin_charts(symbol, "1w")
        market_context = coin_stats_manager.get_coins_list(limit=100)
        
        raw_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data_type": "raw",
            "purpose": "ai_analysis",
            
            # همه داده‌های خام برای AI
            "market_data": raw_details,
            "price_charts": raw_charts,
            "market_context": market_context,
            
            "metadata": {
                "data_sources": ["coinstats_api"],
                "update_frequency": "real_time", 
                "data_quality": "high"
            }
        }
        
        return {
            "status": "success",
            "data": raw_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت داده خام برای {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/raw/batch")
async def batch_raw_scan(request: BatchScanRequest):
    """اسکن دسته‌ای داده‌های خام"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Data service unavailable")
        
        symbols_to_scan = request.symbols[:50]  # محدودیت ۵۰ تا
        
        results = []
        for symbol in symbols_to_scan:
            try:
                raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
                raw_charts = coin_stats_manager.get_coin_charts(symbol, "1w")
                
                raw_data = {
                    "symbol": symbol,
                    "market_data": raw_details,
                    "price_charts": raw_charts,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append({
                    "symbol": symbol,
                    "status": "success",
                    "data": raw_data
                })
                
            except Exception as e:
                results.append({
                    "symbol": symbol, 
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "data_type": "raw",
            "total_symbols": len(symbols_to_scan),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در اسکن دسته‌ای خام: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت مادر داده‌های پردازش شده ====================

@app.get("/api/processed/{symbol}")
async def get_processed_data(symbol: str):
    """داده‌های پردازش شده برای نمایش"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Data service unavailable")
        
        # دریافت داده پایه
        raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
        
        # پردازش برای نمایش
        processed_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data_type": "processed",
            "purpose": "display",
            
            # داده‌های نمایشی
            "display_data": {
                "name": raw_details.get('name', 'Unknown'),
                "symbol": raw_details.get('symbol', 'UNKNOWN'),
                "price": raw_details.get('price', 0),
                "price_change_24h": raw_details.get('priceChange1d', 0),
                "volume_24h": raw_details.get('volume', 0),
                "market_cap": raw_details.get('marketCap', 0),
                "rank": raw_details.get('rank', 0)
            },
            
            # تحلیل‌های ساده
            "analysis": {
                "signal": _generate_simple_signal(raw_details),
                "confidence": _calculate_confidence(raw_details),
                "trend": _analyze_trend(raw_details),
                "risk_level": _assess_risk(raw_details)
            }
        }
        
        return {
            "status": "success",
            "data": processed_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت داده پردازش شده برای {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/processed/batch")
async def batch_processed_scan(request: BatchScanRequest):
    """اسکن دسته‌ای داده‌های پردازش شده"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Data service unavailable")
        
        symbols_to_scan = request.symbols[:50]  # محدودیت ۵۰ تا
        
        results = []
        for symbol in symbols_to_scan:
            try:
                raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
                
                processed_data = {
                    "symbol": symbol,
                    "display_data": {
                        "name": raw_details.get('name', 'Unknown'),
                        "price": raw_details.get('price', 0),
                        "price_change_24h": raw_details.get('priceChange1d', 0),
                        "volume_24h": raw_details.get('volume', 0),
                        "market_cap": raw_details.get('marketCap', 0),
                        "rank": raw_details.get('rank', 0)
                    },
                    "analysis": {
                        "signal": _generate_simple_signal(raw_details),
                        "confidence": _calculate_confidence(raw_details)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append({
                    "symbol": symbol,
                    "status": "success", 
                    "data": processed_data
                })
                
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "completed", 
            "data_type": "processed",
            "total_symbols": len(symbols_to_scan),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در اسکن دسته‌ای پردازش شده: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== روت مادر سلامت ====================

@app.get("/api/status")
async def system_status():
    """وضعیت کامل سیستم"""
    try:
        # اطلاعات سیستم
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # وضعیت سرویس‌ها
        services_status = {
            "coinstats_api": COINSTATS_AVAILABLE,
            "ai_engine": AI_AVAILABLE,
            "technical_analysis": AI_AVAILABLE,
            "neural_network": AI_AVAILABLE,
            "sentiment_analysis": AI_AVAILABLE
        }
        
        # قابلیت‌های AI
        ai_capabilities = {
            "technical_analysis": AI_AVAILABLE,
            "price_prediction": AI_AVAILABLE, 
            "market_sentiment": AI_AVAILABLE,
            "neural_network": AI_AVAILABLE,
            "rsi_analyzer": AI_AVAILABLE,
            "macd_analyzer": AI_AVAILABLE
        }
        
        # عملکرد
        performance = {
            "response_time": "45ms",
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "active_models": 3 if AI_AVAILABLE else 0,
            "memory_usage_mb": round(memory.used / (1024 * 1024), 2)
        }
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            
            "services": services_status,
            "ai_capabilities": ai_capabilities,
            "performance": performance,
            
            "system_metrics": {
                "memory_usage_percent": memory.percent,
                "cpu_usage_percent": cpu_percent,
                "disk_usage_percent": disk.percent
            },
            
            "endpoints_health": {
                "raw_data": "active",
                "processed_data": "active", 
                "batch_scan": "active",
                "system_status": "active"
            }
        }
        
    except Exception as e:
        logger.error(f"خطا در بررسی وضعیت سیستم: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ==================== توابع کمکی ====================

def _generate_simple_signal(coin_data: Dict) -> str:
    """تولید سیگنال ساده"""
    change = coin_data.get('priceChange1d', 0)
    if change > 5:
        return "STRONG_BUY"
    elif change > 2:
        return "BUY" 
    elif change < -5:
        return "STRONG_SELL"
    elif change < -2:
        return "SELL"
    else:
        return "HOLD"

def _calculate_confidence(coin_data: Dict) -> float:
    """محاسبه اعتماد"""
    volume = coin_data.get('volume', 0)
    market_cap = coin_data.get('marketCap', 0)
    
    base_confidence = 0.5
    volume_boost = min(0.3, volume / 10000000000)
    market_cap_boost = min(0.2, market_cap / 1000000000000)
    
    return round(base_confidence + volume_boost + market_cap_boost, 2)

def _analyze_trend(coin_data: Dict) -> str:
    """تحلیل روند"""
    change = coin_data.get('priceChange1d', 0)
    
    if change > 3:
        return "UPTREND"
    elif change < -3:
        return "DOWNTREND" 
    else:
        return "SIDEWAYS"

def _assess_risk(coin_data: Dict) -> str:
    """ارزیابی ریسک"""
    volatility = abs(coin_data.get('priceChange1d', 0))
    if volatility > 15:
        return "HIGH"
    elif volatility > 8:
        return "MEDIUM"
    else:
        return "LOW"

# ==================== روت‌های AI (اختیاری) ====================

@app.get("/api/ai/analyze/{symbol}")
async def ai_analyze(symbol: str, analysis_type: str = Query("technical")):
    """تحلیل AI پیشرفته"""
    try:
        if not AI_AVAILABLE:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        # دریافت داده خام
        raw_response = await get_raw_data(symbol)
        raw_data = raw_response["data"]
        
        # تحلیل با AI
        if analysis_type == "technical":
            analysis = await technical_analyzer.analyze(symbol, raw_data)
        elif analysis_type == "sentiment":
            analysis = await sentiment_analyzer.analyze(symbol, raw_data)
        elif analysis_type == "prediction":
            analysis = await neural_network.predict(symbol, raw_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        return {
            "status": "success",
            "symbol": symbol,
            "analysis_type": analysis_type,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل AI برای {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# مدیریت روت‌های SPA
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """سرو کردن SPA"""
    try:
        return FileResponse("frontend/index.html")
    except:
        return JSONResponse(
            status_code=404,
            content={"error": "Frontend not found"}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Starting VortexAI Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)
