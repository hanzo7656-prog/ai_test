# main.py - با اندپوینت‌های هیبریدی خام/پردازش شده
from fastapi import FastAPI, HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CryptoAI Hybrid API", version="1.0.0")

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

class HybridScanRequest(BaseModel):
    symbols: List[str]
    data_type: str = "processed"  # raw, processed, hybrid
    include_analysis: bool = True

# ایمپورت مدیر CoinStats
try:
    from complete_coinstats_manager import coin_stats_manager
    COINSTATS_AVAILABLE = True
    logger.info("✅ CoinStats Manager loaded successfully")
except ImportError as e:
    COINSTATS_AVAILABLE = False
    logger.error(f"❌ CoinStats Manager import failed: {e}")

# ==================== پردازشگر داده‌های خام ====================

class DataProcessor:
    """پردازشگر داده‌های خام به فرمت‌های مختلف"""
    
    @staticmethod
    def get_raw_data(symbol: str) -> Dict[str, Any]:
        """دریافت داده خام برای AI"""
        try:
            # دریافت داده‌های خام از CoinStats
            raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
            raw_charts = coin_stats_manager.get_coin_charts(symbol, "1w")
            raw_market = coin_stats_manager.get_coins_list(limit=100)
            
            return {
                "data_type": "raw",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "raw_details": raw_details,
                "raw_charts": raw_charts,
                "market_context": raw_market,
                "data_structure": {
                    "details_keys": list(raw_details.keys()) if raw_details else [],
                    "charts_keys": list(raw_charts.keys()) if raw_charts else [],
                    "market_keys": list(raw_market.keys()) if raw_market else []
                }
            }
        except Exception as e:
            logger.error(f"خطا در دریافت داده خام {symbol}: {e}")
            return {
                "data_type": "raw",
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_processed_data(symbol: str) -> Dict[str, Any]:
        """پردازش داده برای نمایش معمولی"""
        try:
            raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
            
            if not raw_details or 'result' not in raw_details:
                return {
                    "success": False,
                    "error": "داده‌ای دریافت نشد",
                    "symbol": symbol
                }
            
            coin_data = raw_details['result']
            
            # پردازش برای نمایش کاربرپسند
            processed = {
                "data_type": "processed",
                "success": True,
                "symbol": symbol,
                "display_data": {
                    "name": coin_data.get('name', 'Unknown'),
                    "price": f"${coin_data.get('price', 0):,.2f}",
                    "price_change_24h": f"{coin_data.get('priceChange1d', 0):+.2f}%",
                    "volume_24h": f"${coin_data.get('volume', 0):,.0f}",
                    "market_cap": f"${coin_data.get('marketCap', 0):,.0f}",
                    "rank": f"#{coin_data.get('rank', 0)}",
                    "high_24h": f"${coin_data.get('high', 0):,.2f}",
                    "low_24h": f"${coin_data.get('low', 0):,.2f}"
                },
                "analysis": {
                    "signal": DataProcessor._generate_signal(coin_data),
                    "confidence": DataProcessor._calculate_confidence(coin_data),
                    "trend": DataProcessor._analyze_trend(coin_data),
                    "risk_level": DataProcessor._assess_risk(coin_data)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"خطا در پردازش داده {symbol}: {e}")
            return {
                "data_type": "processed", 
                "success": False,
                "error": str(e),
                "symbol": symbol
            }
    
    @staticmethod
    def get_hybrid_data(symbol: str) -> Dict[str, Any]:
        """داده هیبریدی - هم خام هم پردازش شده"""
        raw_data = DataProcessor.get_raw_data(symbol)
        processed_data = DataProcessor.get_processed_data(symbol)
        
        return {
            "data_type": "hybrid",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "raw_data": raw_data,
            "processed_data": processed_data,
            "summary": {
                "raw_available": "error" not in raw_data,
                "processed_available": processed_data.get("success", False),
                "data_quality": "good" if "error" not in raw_data and processed_data.get("success") else "poor"
            }
        }
    
    @staticmethod
    def _generate_signal(coin_data: Dict) -> str:
        """تولید سیگنال ساده"""
        change = coin_data.get('priceChange1d', 0)
        if change > 3:
            return "BUY"
        elif change < -3:
            return "SELL"
        else:
            return "HOLD"
    
    @staticmethod
    def _calculate_confidence(coin_data: Dict) -> float:
        """محاسبه اعتماد"""
        volume = coin_data.get('volume', 0)
        change = abs(coin_data.get('priceChange1d', 0))
        
        base_confidence = 0.5
        volume_boost = min(0.3, volume / 1000000000)  # نرمال‌سازی حجم
        change_boost = min(0.2, change / 20)  # نرمال‌سازی تغییرات
        
        return round(base_confidence + volume_boost + change_boost, 2)
    
    @staticmethod
    def _analyze_trend(coin_data: Dict) -> str:
        """تحلیل روند"""
        change = coin_data.get('priceChange1d', 0)
        if change > 2:
            return "صعودی"
        elif change < -2:
            return "نزولی"
        else:
            return "خنثی"
    
    @staticmethod
    def _assess_risk(coin_data: Dict) -> str:
        """ارزیابی ریسک"""
        volatility = abs(coin_data.get('priceChange1d', 0))
        if volatility > 10:
            return "بالا"
        elif volatility > 5:
            return "متوسط"
        else:
            return "پایین"

# ==================== سیستم اسکن چندحالته ====================

class HybridScanEngine:
    """موتور اسکن با قابلیت چندحالته"""
    
    def __init__(self):
        self.scan_count = 0
    
    def scan_basic(self, symbols: List[str]) -> Dict[str, Any]:
        """اسکن معمولی - فقط داده پردازش شده"""
        self.scan_count += 1
        logger.info(f"🔍 اسکن معمولی برای {len(symbols)} نماد")
        
        results = []
        for symbol in symbols:
            processed_data = DataProcessor.get_processed_data(symbol)
            results.append(processed_data)
        
        return {
            "scan_type": "basic",
            "data_type": "processed", 
            "results": results,
            "summary": {
                "total": len(symbols),
                "successful": len([r for r in results if r.get('success')]),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def scan_ai_ready(self, symbols: List[str]) -> Dict[str, Any]:
        """اسکن مخصوص AI - داده خام"""
        self.scan_count += 1
        logger.info(f"🤖 اسکن AI برای {len(symbols)} نماد")
        
        results = []
        for symbol in symbols:
            raw_data = DataProcessor.get_raw_data(symbol)
            results.append(raw_data)
        
        return {
            "scan_type": "ai_ready",
            "data_type": "raw",
            "results": results,
            "summary": {
                "total": len(symbols),
                "raw_data_quality": f"{len([r for r in results if 'error' not in r])}/{len(results)}",
                "ai_compatible": True,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def scan_hybrid(self, symbols: List[str]) -> Dict[str, Any]:
        """اسکن هیبریدی - هر دو نوع داده"""
        self.scan_count += 1
        logger.info(f"🔀 اسکن هیبریدی برای {len(symbols)} نماد")
        
        results = []
        for symbol in symbols:
            hybrid_data = DataProcessor.get_hybrid_data(symbol)
            results.append(hybrid_data)
        
        return {
            "scan_type": "hybrid", 
            "data_type": "hybrid",
            "results": results,
            "summary": {
                "total": len(symbols),
                "raw_available": len([r for r in results if r.get('summary', {}).get('raw_available')]),
                "processed_available": len([r for r in results if r.get('summary', {}).get('processed_available')]),
                "timestamp": datetime.now().isoformat()
            }
        }

# ایجاد موتور اسکن
scan_engine = HybridScanEngine()

# ==================== روت‌های API ====================

api_router = APIRouter(prefix="/api")

@api_router.get("/health")
async def health_check():
    """سلامت سیستم"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "coinstats_available": COINSTATS_AVAILABLE,
        "total_scans": scan_engine.scan_count,
        "features": ["basic_scan", "ai_scan", "hybrid_scan", "raw_data", "processed_data"]
    }

# ==================== اندپوینت‌های اسکن چندحالته ====================

@api_router.post("/scan/basic")
async def basic_scan(request: ScanRequest):
    """اسکن معمولی - برای فرانت‌اند"""
    try:
        results = scan_engine.scan_basic(request.symbols)
        return {
            "status": "success",
            "scan_mode": "basic",
            "data_type": "processed",
            **results
        }
    except Exception as e:
        logger.error(f"خطا در اسکن معمولی: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/scan/ai")
async def ai_scan(request: ScanRequest):
    """اسکن مخصوص AI - داده خام"""
    try:
        results = scan_engine.scan_ai_ready(request.symbols)
        return {
            "status": "success", 
            "scan_mode": "ai",
            "data_type": "raw",
            "ai_compatible": True,
            **results
        }
    except Exception as e:
        logger.error(f"خطا در اسکن AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/scan/hybrid")
async def hybrid_scan(request: HybridScanRequest):
    """اسکن هیبریدی - هر دو نوع داده"""
    try:
        results = scan_engine.scan_hybrid(request.symbols)
        return {
            "status": "success",
            "scan_mode": "hybrid",
            "data_type": "hybrid",
            **results
        }
    except Exception as e:
        logger.error(f"خطا در اسکن هیبریدی: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== اندپوینت‌های دسترسی مستقیم به داده ====================

@api_router.get("/data/raw/{symbol}")
async def get_raw_data(symbol: str):
    """دریافت داده خام برای AI"""
    try:
        raw_data = DataProcessor.get_raw_data(symbol)
        return {
            "status": "success",
            "data_type": "raw",
            "ai_compatible": True,
            "symbol": symbol,
            "data": raw_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/processed/{symbol}")
async def get_processed_data(symbol: str):
    """دریافت داده پردازش شده برای نمایش"""
    try:
        processed_data = DataProcessor.get_processed_data(symbol)
        return {
            "status": "success" if processed_data.get('success') else "error",
            "data_type": "processed",
            "symbol": symbol,
            "data": processed_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/hybrid/{symbol}")
async def get_hybrid_data(symbol: str):
    """دریافت داده هیبریدی"""
    try:
        hybrid_data = DataProcessor.get_hybrid_data(symbol)
        return {
            "status": "success",
            "data_type": "hybrid", 
            "symbol": symbol,
            "data": hybrid_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== اندپوینت‌های کمکی ====================

@api_router.get("/system/status")
async def system_status():
    """وضعیت سیستم"""
    return {
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "available_endpoints": [
            "POST /api/scan/basic - اسکن معمولی",
            "POST /api/scan/ai - اسکن AI (داده خام)",
            "POST /api/scan/hybrid - اسکن هیبریدی",
            "GET /api/data/raw/{symbol} - داده خام",
            "GET /api/data/processed/{symbol} - داده پردازش شده",
            "GET /api/data/hybrid/{symbol} - داده هیبریدی"
        ]
    }

# ثبت روت‌ها
app.include_router(api_router)

# ==================== مدیریت عمومی ====================

@app.get("/")
async def root():
    return {
        "message": "CryptoAI Hybrid API",
        "status": "running", 
        "timestamp": datetime.now().isoformat(),
        "documentation": "از اندپوینت‌های /api استفاده کنید"
    }

@app.get("/{path:path}")
async def catch_all(path: str):
    if path.startswith('api/'):
        raise HTTPException(status_code=404, detail="Endpoint not found")
    try:
        return FileResponse("frontend/index.html")
    except:
        return JSONResponse(
            status_code=404,
            content={"error": "Frontend not found"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
