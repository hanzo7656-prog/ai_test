# main.py - سرور اصلی VortexAI با سیستم اسکن دسته‌ای
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import time
import psutil
from pathlib import Path

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VortexAI API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ایمپورت مدیر CoinStats
try:
    from complete_coinstats_manager import coin_stats_manager
    COINSTATS_AVAILABLE = True
    logger.info("✅ CoinStats Manager loaded successfully")
except ImportError as e:
    COINSTATS_AVAILABLE = False
    logger.error(f"❌ CoinStats Manager import failed: {e}")

# ایمپورت سیستم GitHub DB
try:
    from github_db import GitHubDBCache, BatchScanner, DataCompressor, ProgressTracker
    GITHUB_DB_AVAILABLE = True
    logger.info("✅ GitHub DB System loaded successfully")
    
    # ایجاد نمونه‌ها
    github_cache = GitHubDBCache()
    batch_scanner = BatchScanner(github_cache)
    data_compressor = DataCompressor()
    progress_tracker = ProgressTracker()
    
except ImportError as e:
    GITHUB_DB_AVAILABLE = False
    logger.error(f"❌ GitHub DB import failed: {e}")

# مدل‌های درخواست
class ScanRequest(BaseModel):
    symbols: List[str]
    limit: Optional[int] = 100

class MultiScanRequest(BaseModel):
    symbols: List[str]
    scan_type: str = "basic"
    limit: Optional[int] = 100

# پردازشگر داده‌ها
class DataProcessor:
    """پردازشگر داده‌های کوین"""
    
    @staticmethod
    def get_ai_scan_data(symbol: str, limit: int = 500) -> Dict[str, Any]:
        """داده خام برای هوش مصنوعی تحلیلگر تکنیکال"""
        try:
            start_time = time.time()
            
            # دریافت داده‌های خام از API
            raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
            raw_charts = coin_stats_manager.get_coin_charts(symbol, "1w")
            market_context = coin_stats_manager.get_coins_list(limit=min(limit, 1000))
            
            response_time = round((time.time() - start_time) * 1000, 2)
            
            # ساختار داده خام برای AI
            ai_data = {
                "data_type": "raw",
                "purpose": "ai_technical_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time,
                
                # داده‌های خام اصلی
                "raw_data": {
                    "coin_details": raw_details,
                    "price_charts": raw_charts,
                    "market_context": market_context
                },
                
                # متادیتای فنی
                "technical_metadata": {
                    "data_sources": ["coinstats_api"],
                    "update_frequency": "real_time",
                    "data_quality": "high",
                    "fields_available": list(raw_details.keys()) if isinstance(raw_details, dict) else []
                }
            }
            
            return ai_data
            
        except Exception as e:
            logger.error(f"خطا در دریافت داده AI برای {symbol}: {e}")
            return {
                "data_type": "raw",
                "purpose": "ai_technical_analysis", 
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_basic_scan_data(symbol: str, limit: int = 100) -> Dict[str, Any]:
        """داده پردازش شده برای نمایش معمولی"""
        try:
            start_time = time.time()
            
            # دریافت داده خام
            raw_details = coin_stats_manager.get_coin_details(symbol, "USD")
            
            # بررسی خطا
            if isinstance(raw_details, dict) and "error" in raw_details:
                return {
                    "success": False,
                    "error": raw_details["error"],
                    "symbol": symbol
                }
            
            # اگر داده یک لیست باشد (ساختار غیرمنتظره)
            if isinstance(raw_details, list):
                if len(raw_details) > 0:
                    coin_data = raw_details[0]  # اولین آیتم را بگیر
                else:
                    return {
                        "success": False,
                        "error": "داده‌ای دریافت نشد",
                        "symbol": symbol
                    }
            else:
                coin_data = raw_details  # مستقیماً استفاده کن
            
            response_time = round((time.time() - start_time) * 1000, 2)
            
            # پردازش برای نمایش کاربرپسند
            processed_data = {
                "data_type": "processed",
                "purpose": "basic_display",
                "success": True,
                "symbol": symbol,
                "response_time_ms": response_time,
                "timestamp": datetime.now().isoformat(),
                
                # داده‌های نمایشی
                "display_data": {
                    "name": coin_data.get('name', 'Unknown'),
                    "symbol": coin_data.get('symbol', 'UNKNOWN'),
                    "price": coin_data.get('price', 0),
                    "price_formatted": f"${coin_data.get('price', 0):,.2f}",
                    "price_change_24h": coin_data.get('priceChange1d', 0),
                    "price_change_24h_formatted": f"{coin_data.get('priceChange1d', 0):+.2f}%",
                    "volume_24h": coin_data.get('volume', 0),
                    "volume_24h_formatted": f"${coin_data.get('volume', 0):,.0f}",
                    "market_cap": coin_data.get('marketCap', 0),
                    "market_cap_formatted": f"${coin_data.get('marketCap', 0):,.0f}",
                    "rank": coin_data.get('rank', 0),
                    "rank_formatted": f"#{coin_data.get('rank', 0)}"
                },
                
                # تحلیل‌های ساده
                "analysis": {
                    "signal": DataProcessor._generate_signal(coin_data),
                    "confidence": DataProcessor._calculate_confidence(coin_data),
                    "trend": DataProcessor._analyze_trend(coin_data),
                    "risk_level": DataProcessor._assess_risk(coin_data),
                    "volatility": DataProcessor._calculate_volatility(coin_data)
                },
                
                # اطلاعات اضافی
                "metadata": {
                    "website": coin_data.get('websiteUrl'),
                    "social_links": {
                        "twitter": coin_data.get('twitterUrl'),
                        "reddit": coin_data.get('redditUrl')
                    }
                }
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"خطا در پردازش داده {symbol}: {e}")
            return {
                "data_type": "processed",
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def _generate_signal(coin_data: Dict) -> str:
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
    
    @staticmethod
    def _calculate_confidence(coin_data: Dict) -> float:
        """محاسبه اعتماد"""
        volume = coin_data.get('volume', 0)
        market_cap = coin_data.get('marketCap', 0)
        
        base_confidence = 0.5
        volume_boost = min(0.3, volume / 10000000000)  # نرمال‌سازی حجم
        market_cap_boost = min(0.2, market_cap / 1000000000000)  # نرمال‌سازی مارکت کپ
        
        return round(base_confidence + volume_boost + market_cap_boost, 2)
    
    @staticmethod
    def _analyze_trend(coin_data: Dict) -> str:
        """تحلیل روند"""
        change_1h = coin_data.get('priceChange1h', 0)
        change_1d = coin_data.get('priceChange1d', 0)
        
        if change_1d > 3 and change_1h > 0:
            return "STRONG_UPTREND"
        elif change_1d > 0:
            return "UPTREND"
        elif change_1d < -3 and change_1h < 0:
            return "STRONG_DOWNTREND"
        elif change_1d < 0:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def _assess_risk(coin_data: Dict) -> str:
        """ارزیابی ریسک"""
        volatility = abs(coin_data.get('priceChange1d', 0))
        if volatility > 15:
            return "VERY_HIGH"
        elif volatility > 8:
            return "HIGH"
        elif volatility > 4:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def _calculate_volatility(coin_data: Dict) -> float:
        """محاسبه نوسان"""
        changes = [
            abs(coin_data.get('priceChange1h', 0)),
            abs(coin_data.get('priceChange1d', 0)),
            abs(coin_data.get('priceChange1w', 0))
        ]
        return round(sum(changes) / len(changes), 2)

# ==================== روت‌های اسکن دسته‌ای جدید ====================

@app.post("/api/scan/batch/raw")
async def batch_scan_raw(
    request: ScanRequest,
    background_tasks: BackgroundTasks
):
    """اسکن دسته‌ای خام - برای هوش مصنوعی"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="CoinStats service unavailable")
        
        # بررسی سایز دسته
        batch_size = min(request.limit, 25)  # حداکثر 25 تایی
        symbols_to_scan = request.symbols[:request.limit]
        
        # شروع اسکن در پس‌زمینه
        background_tasks.add_task(
            process_raw_batch_scan,
            symbols_to_scan,
            batch_size
        )
        
        return {
            "status": "started",
            "scan_type": "raw",
            "total_symbols": len(symbols_to_scan),
            "batch_size": batch_size,
            "message": "اسکن دسته‌ای خام شروع شد",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در شروع اسکن دسته‌ای خام: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan/batch/processed")
async def batch_scan_processed(
    request: ScanRequest,
    background_tasks: BackgroundTasks
):
    """اسکن دسته‌ای پردازش شده - برای نمایش"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="CoinStats service unavailable")
        
        batch_size = min(request.limit, 25)
        symbols_to_scan = request.symbols[:request.limit]
        
        background_tasks.add_task(
            process_processed_batch_scan,
            symbols_to_scan,
            batch_size
        )
        
        return {
            "status": "started", 
            "scan_type": "processed",
            "total_symbols": len(symbols_to_scan),
            "batch_size": batch_size,
            "message": "اسکن دسته‌ای پردازش شده شروع شد",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در شروع اسکن دسته‌ای پردازش شده: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan/progress")
async def get_scan_progress():
    """دریافت پیشرفت اسکن‌های جاری"""
    try:
        if not GITHUB_DB_AVAILABLE:
            return {"status": "github_db_unavailable"}
        
        progress = progress_tracker.get_progress()
        cache_stats = github_cache.get_cache_stats()
        
        return {
            "status": "success",
            "progress": progress,
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت پیشرفت: {e}")
        return {"status": "error", "error": str(e)}

# ==================== توابع پردازش دسته‌ای ====================

async def process_raw_batch_scan(symbols: List[str], batch_size: int):
    """پردازش اسکن دسته‌ای خام"""
    try:
        logger.info(f"🚀 شروع اسکن دسته‌ای خام برای {len(symbols)} ارز")
        
        # بروزرسانی پیشرفت
        progress_tracker.update_progress(
            total_symbols=len(symbols),
            scanned=0,
            current_batch=0,
            status="running_raw"
        )
        
        # تقسیم به دسته‌ها
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        for batch_num, batch_symbols in enumerate(batches):
            batch_results = []
            
            for symbol in batch_symbols:
                try:
                    # دریافت داده خام
                    raw_data = DataProcessor.get_ai_scan_data(symbol)
                    
                    # ذخیره در GitHub DB
                    github_cache.save_live_data(symbol, {
                        "scan_type": "raw",
                        "data": raw_data,
                        "batch_number": batch_num + 1
                    })
                    
                    batch_results.append({
                        "symbol": symbol,
                        "status": "success",
                        "data_type": "raw"
                    })
                    
                except Exception as e:
                    batch_results.append({
                        "symbol": symbol,
                        "status": "error", 
                        "error": str(e)
                    })
            
            # بروزرسانی پیشرفت
            current_scanned = (batch_num * batch_size) + len(batch_symbols)
            progress_tracker.update_progress(
                total_symbols=len(symbols),
                scanned=current_scanned,
                current_batch=batch_num + 1,
                status="running_raw"
            )
            
            logger.info(f"✅ دسته {batch_num + 1} کامل شد: {len(batch_symbols)} ارز")
            
        # تکمیل اسکن
        progress_tracker.update_progress(
            total_symbols=len(symbols),
            scanned=len(symbols),
            current_batch=len(batches),
            status="completed_raw"
        )
        
        logger.info(f"🎉 اسکن دسته‌ای خام کامل شد: {len(symbols)} ارز")
        
    except Exception as e:
        logger.error(f"❌ خطا در اسکن دسته‌ای خام: {e}")
        progress_tracker.update_progress(
            total_symbols=len(symbols),
            scanned=0,
            current_batch=0,
            status="error"
        )

async def process_processed_batch_scan(symbols: List[str], batch_size: int):
    """پردازش اسکن دسته‌ای پردازش شده"""
    try:
        logger.info(f"🚀 شروع اسکن دسته‌ای پردازش شده برای {len(symbols)} ارز")
        
        progress_tracker.update_progress(
            total_symbols=len(symbols),
            scanned=0,
            current_batch=0,
            status="running_processed"
        )
        
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        for batch_num, batch_symbols in enumerate(batches):
            batch_results = []
            
            for symbol in batch_symbols:
                try:
                    # دریافت داده پردازش شده
                    processed_data = DataProcessor.get_basic_scan_data(symbol)
                    
                    # ذخیره در GitHub DB
                    github_cache.save_live_data(symbol, {
                        "scan_type": "processed", 
                        "data": processed_data,
                        "batch_number": batch_num + 1
                    })
                    
                    batch_results.append({
                        "symbol": symbol,
                        "status": "success",
                        "data_type": "processed"
                    })
                    
                except Exception as e:
                    batch_results.append({
                        "symbol": symbol,
                        "status": "error",
                        "error": str(e)
                    })
            
            current_scanned = (batch_num * batch_size) + len(batch_symbols)
            progress_tracker.update_progress(
                total_symbols=len(symbols),
                scanned=current_scanned,
                current_batch=batch_num + 1, 
                status="running_processed"
            )
            
            logger.info(f"✅ دسته {batch_num + 1} کامل شد: {len(batch_symbols)} ارز")
            
        progress_tracker.update_progress(
            total_symbols=len(symbols),
            scanned=len(symbols),
            current_batch=len(batches),
            status="completed_processed"
        )
        
        logger.info(f"🎉 اسکن دسته‌ای پردازش شده کامل شد: {len(symbols)} ارز")
        
    except Exception as e:
        logger.error(f"❌ خطا در اسکن دسته‌ای پردازش شده: {e}")
        progress_tracker.update_progress(
            total_symbols=len(symbols),
            scanned=0, 
            current_batch=0,
            status="error"
        )

# ==================== روت‌های اصلی API ====================

@app.get("/")
async def root():
    """صفحه اصلی - سرو کردن frontend"""
    try:
        return FileResponse("frontend/index.html")
    except:
        return JSONResponse(
            status_code=200,
            content={
                "message": "VortexAI API Server",
                "status": "running",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "ai_scan": "GET /api/scan/ai/{symbol}",
                    "basic_scan": "GET /api/scan/basic/{symbol}",
                    "batch_raw": "POST /api/scan/batch/raw",
                    "batch_processed": "POST /api/scan/batch/processed",
                    "scan_progress": "GET /api/scan/progress",
                    "system_status": "GET /api/system/status",
                    "clear_cache": "GET /api/debug/clear-cache"
                }
            }
        )

@app.get("/api/scan/ai/{symbol}")
async def ai_scan(
    symbol: str,
    limit: int = Query(500, ge=1, le=1000, description="تعداد داده‌های درخواستی")
):
    """اسکن مخصوص هوش مصنوعی تحلیلگر تکنیکال - داده خام"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="CoinStats service unavailable")
        
        ai_data = DataProcessor.get_ai_scan_data(symbol.lower(), limit)
        
        return {
            "status": "success" if "error" not in ai_data else "error",
            "data_type": "raw",
            "purpose": "ai_technical_analysis",
            "symbol": symbol,
            "limit": limit,
            "data": ai_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در اسکن AI برای {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan/basic/{symbol}")
async def basic_scan(
    symbol: str,
    limit: int = Query(100, ge=1, le=500, description="تعداد داده‌های درخواستی")
):
    """اسکن معمولی برای نمایش - داده پردازش شده"""
    try:
        if not COINSTATS_AVAILABLE:
            raise HTTPException(status_code=503, detail="CoinStats service unavailable")
        
        basic_data = DataProcessor.get_basic_scan_data(symbol.lower(), limit)
        
        return {
            "status": "success" if basic_data.get("success") else "error",
            "data_type": "processed", 
            "purpose": "basic_display",
            "symbol": symbol,
            "limit": limit,
            "data": basic_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطا در اسکن معمولی برای {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def system_status():
    """وضعیت کامل سیستم و سلامت سرویس‌ها"""
    try:
        # تست سلامت اندپوینت‌های API
        endpoint_health = {}
        if COINSTATS_AVAILABLE:
            endpoint_health = coin_stats_manager.test_all_endpoints()
        
        # متریک‌های سیستم
        system_metrics = {}
        if COINSTATS_AVAILABLE:
            system_metrics = coin_stats_manager.get_system_metrics()
        
        # آمار کش - نسخه اصلاح شده
        cache_files = []
        cache_size = 0
        if os.path.exists("./coinstats_cache"):
            cache_files = [f for f in os.listdir("./coinstats_cache") if f.endswith('.json')]
            for file in cache_files:
                file_path = os.path.join("./coinstats_cache", file)
                cache_size += os.path.getsize(file_path)
        
        # اطلاعات سیستم
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            
            "services": {
                "coinstats_api": COINSTATS_AVAILABLE,
                "github_db": GITHUB_DB_AVAILABLE,
                "total_healthy_endpoints": sum(1 for r in endpoint_health.values() if r.get('status') == 'success'),
                "total_endpoints": len(endpoint_health)
            },
            
            "endpoints_health": endpoint_health,
            
            "system_metrics": {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent
                },
                "cpu": {
                    "percent": cpu_percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": disk.percent
                }
            },
            
            "cache": {
                "total_files": len(cache_files),
                "total_size_mb": round(cache_size / (1024 * 1024), 2),
                "cache_dir": "./coinstats_cache"
            },
            
            "github_db_stats": github_cache.get_cache_stats() if GITHUB_DB_AVAILABLE else {},
            
            "usage_stats": {
                "active_connections": 0,
                "uptime_seconds": int(time.time() - psutil.boot_time())
            }
        }
        
    except Exception as e:
        logger.error(f"خطا در بررسی وضعیت سیستم: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/debug/clear-cache")
async def clear_cache():
    """پاک کردن کش (فقط برای دیباگ)"""
    try:
        if COINSTATS_AVAILABLE:
            coin_stats_manager.clear_cache()
            return {
                "status": "success", 
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error", 
                "message": "CoinStats not available",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# مدیریت روت‌های SPA برای frontend
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """سرو کردن frontend برای تمام مسیرها"""
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
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)
