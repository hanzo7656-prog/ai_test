# main.py - نسخه ساده و متمرکز روی اسکن
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import traceback

# تنظیمات پیشرفته لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CryptoAI Scan API", version="1.0.0")

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

# مدل درخواست اسکن
class ScanRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1h"
    scan_mode: str = "ai"

# ==================== مدیریت خطا و لاگینگ ====================

class ErrorHandler:
    """مدیریت پیشرفته خطاها"""
    
    @staticmethod
    def log_error(operation: str, error: Exception, details: Dict = None):
        """لاگ کردن خطا با جزئیات کامل"""
        error_details = {
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        if details:
            error_details.update(details)
        
        logger.error(f"❌ {operation} failed: {error}")
        logger.debug(f"🔍 Error details: {error_details}")
        
        return error_details

# ==================== اتصال به CoinStats API ====================

class CoinStatsManager:
    """مدیریت اتصال به CoinStats API با مدیریت خطای کامل"""
    
    def __init__(self):
        self.initialized = False
        self.api_status = "unknown"
        self.last_error = None
        self.coin_stats_manager = None
        
        self._initialize()
    
    def _initialize(self):
        """راه‌اندازی اتصال به CoinStats"""
        try:
            logger.info("🔄 در حال راه‌اندازی اتصال به CoinStats API...")
            
            from complete_coinstats_manager import coin_stats_manager
            self.coin_stats_manager = coin_stats_manager
            self.initialized = True
            
            # تست اتصال
            test_result = self._test_connection()
            if test_result:
                self.api_status = "connected"
                logger.info("✅ اتصال به CoinStats API با موفقیت برقرار شد")
            else:
                self.api_status = "connection_failed"
                logger.error("❌ اتصال به CoinStats API ناموفق بود")
                
        except ImportError as e:
            self.initialized = False
            self.api_status = "import_error"
            self.last_error = str(e)
            logger.error(f"❌ خطای ایمپورت CoinStats: {e}")
            
        except Exception as e:
            self.initialized = False
            self.api_status = "initialization_error"
            self.last_error = str(e)
            ErrorHandler.log_error("CoinStats initialization", e)
    
    def _test_connection(self) -> bool:
        """تست اتصال به API"""
        try:
            if not self.coin_stats_manager:
                return False
                
            # تست دریافت داده
            result = self.coin_stats_manager.get_coins_list(limit=1)
            
            if result and isinstance(result, dict) and 'result' in result:
                coins = result['result']
                if coins and len(coins) > 0:
                    logger.info(f"✅ تست اتصال موفق - داده دریافت شد: {len(coins)} کوین")
                    return True
            
            logger.warning("⚠️ تست اتصال: داده‌ای دریافت نشد")
            return False
            
        except Exception as e:
            ErrorHandler.log_error("API connection test", e)
            return False
    
    def get_coin_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌های کوین با مدیریت خطای کامل"""
        if not self.initialized or not self.coin_stats_manager:
            error_msg = "CoinStats Manager راه‌اندازی نشده است"
            return {
                'success': False,
                'error': error_msg,
                'symbol': symbol,
                'data': None
            }
        
        try:
            logger.info(f"🔍 در حال دریافت داده برای {symbol}...")
            
            # دریافت جزئیات کوین
            details = self.coin_stats_manager.get_coin_details(symbol, "USD")
            
            # دریافت داده‌های چارت
            charts = self.coin_stats_manager.get_coin_charts(symbol, "1w")
            
            # اعتبارسنجی داده‌ها
            if not details or 'result' not in details:
                return {
                    'success': False,
                    'error': 'داده‌ای از API دریافت نشد',
                    'symbol': symbol,
                    'data': None
                }
            
            coin_data = details['result']
            
            # بررسی ساختار داده
            if not isinstance(coin_data, dict):
                return {
                    'success': False,
                    'error': 'ساختار داده نامعتبر است',
                    'symbol': symbol,
                    'data': None
                }
            
            # استخراج اطلاعات مهم
            processed_data = {
                'symbol': symbol,
                'name': coin_data.get('name', 'Unknown'),
                'price': coin_data.get('price', 0),
                'price_change_24h': coin_data.get('priceChange1d', 0),
                'price_change_percent_24h': coin_data.get('priceChange1d', 0),
                'high_24h': coin_data.get('high', 0),
                'low_24h': coin_data.get('low', 0),
                'volume_24h': coin_data.get('volume', 0),
                'market_cap': coin_data.get('marketCap', 0),
                'rank': coin_data.get('rank', 0),
                'website': coin_data.get('websiteUrl', ''),
                'timestamp': datetime.now().isoformat(),
                'raw_data': coin_data  # داده خام برای دیباگ
            }
            
            logger.info(f"✅ داده‌های {symbol} با موفقیت دریافت شد: ${processed_data['price']}")
            
            return {
                'success': True,
                'error': None,
                'symbol': symbol,
                'data': processed_data
            }
            
        except Exception as e:
            error_details = ErrorHandler.log_error(
                f"Get coin data for {symbol}", 
                e,
                {'symbol': symbol}
            )
            
            return {
                'success': False,
                'error': f"خطا در دریافت داده: {str(e)}",
                'symbol': symbol,
                'data': None,
                'debug_info': error_details
            }
    
    def get_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال"""
        return {
            'initialized': self.initialized,
            'api_status': self.api_status,
            'last_error': self.last_error,
            'timestamp': datetime.now().isoformat()
        }

# ایجاد مدیر CoinStats
coin_stats = CoinStatsManager()

# ==================== سیستم اسکن ====================

class ScanEngine:
    """موتور اسکن با مدیریت خطای پیشرفته"""
    
    def __init__(self):
        self.scan_count = 0
        self.successful_scans = 0
        self.failed_scans = 0
    
    async def scan_symbols(self, symbols: List[str], scan_mode: str) -> Dict[str, Any]:
        """اسکن چندنماد با مدیریت خطا"""
        self.scan_count += 1
        logger.info(f"🔍 شروع اسکن برای {len(symbols)} نماد: {symbols}")
        
        results = []
        successful = 0
        failed = 0
        
        for symbol in symbols:
            try:
                # دریافت داده از CoinStats
                coin_result = coin_stats.get_coin_data(symbol)
                
                if coin_result['success']:
                    # تحلیل داده‌ها
                    analysis = self._analyze_coin_data(coin_result['data'])
                    results.append(analysis)
                    successful += 1
                    logger.info(f"✅ اسکن موفق برای {symbol}")
                else:
                    # خطا در دریافت داده
                    error_analysis = {
                        'symbol': symbol,
                        'success': False,
                        'error': coin_result['error'],
                        'price': 0,
                        'change_24h': 0,
                        'volume': 'N/A',
                        'market_cap': 'N/A',
                        'signal': 'ERROR',
                        'confidence': 0,
                        'timestamp': datetime.now().isoformat(),
                        'debug_info': coin_result.get('debug_info')
                    }
                    results.append(error_analysis)
                    failed += 1
                    logger.warning(f"⚠️ اسکن ناموفق برای {symbol}: {coin_result['error']}")
                    
            except Exception as e:
                # خطای غیرمنتظره
                error_details = ErrorHandler.log_error(
                    f"Scan symbol {symbol}", 
                    e,
                    {'symbol': symbol, 'scan_mode': scan_mode}
                )
                
                error_analysis = {
                    'symbol': symbol,
                    'success': False,
                    'error': f"خطای غیرمنتظره: {str(e)}",
                    'price': 0,
                    'change_24h': 0,
                    'volume': 'N/A',
                    'market_cap': 'N/A',
                    'signal': 'ERROR',
                    'confidence': 0,
                    'timestamp': datetime.now().isoformat(),
                    'debug_info': error_details
                }
                results.append(error_analysis)
                failed += 1
                logger.error(f"❌ خطای غیرمنتظره در اسکن {symbol}")
        
        # آپدیت آمار
        self.successful_scans += successful
        self.failed_scans += failed
        
        return {
            'scan_results': results,
            'summary': {
                'total_scanned': len(symbols),
                'successful': successful,
                'failed': failed,
                'success_rate': f"{(successful/len(symbols))*100:.1f}%" if symbols else "0%",
                'scan_mode': scan_mode,
                'timestamp': datetime.now().isoformat()
            },
            'api_status': coin_stats.get_status()
        }
    
    def _analyze_coin_data(self, coin_data: Dict) -> Dict[str, Any]:
        """تحلیل داده‌های کوین"""
        try:
            price = coin_data.get('price', 0)
            change_24h = coin_data.get('price_change_24h', 0)
            
            # تولید سیگنال ساده
            if change_24h > 5:
                signal = "BUY"
                confidence = 0.8
            elif change_24h < -5:
                signal = "SELL"
                confidence = 0.7
            else:
                signal = "HOLD"
                confidence = 0.6
            
            return {
                'symbol': coin_data['symbol'],
                'success': True,
                'price': price,
                'change_24h': change_24h,
                'volume': f"{coin_data.get('volume_24h', 0):,.0f}",
                'market_cap': f"{coin_data.get('market_cap', 0):,.0f}",
                'signal': signal,
                'confidence': confidence,
                'timestamp': coin_data['timestamp'],
                'raw_data_available': True
            }
            
        except Exception as e:
            ErrorHandler.log_error("Analyze coin data", e, {'coin_data': coin_data})
            
            return {
                'symbol': coin_data.get('symbol', 'UNKNOWN'),
                'success': False,
                'error': f"خطا در تحلیل داده: {str(e)}",
                'price': 0,
                'change_24h': 0,
                'volume': 'N/A',
                'market_cap': 'N/A',
                'signal': 'ERROR',
                'confidence': 0,
                'timestamp': datetime.now().isoformat()
            }

# ایجاد موتور اسکن
scan_engine = ScanEngine()

# ==================== روت‌های API ====================

api_router = APIRouter(prefix="/api")

@api_router.get("/health")
async def health_check():
    """بررسی سلامت API"""
    return {
        "status": "healthy",
        "service": "crypto-ai-scan",
        "timestamp": datetime.now().isoformat(),
        "coinstats_status": coin_stats.get_status(),
        "scan_stats": {
            "total_scans": scan_engine.scan_count,
            "successful_scans": scan_engine.successful_scans,
            "failed_scans": scan_engine.failed_scans
        }
    }

@api_router.get("/system/status")
async def system_status():
    """وضعیت سیستم"""
    return {
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": ["scan", "real-time-data", "error-handling"],
        "coinstats_api": coin_stats.get_status()
    }

@api_router.post("/ai/scan")
async def ai_scan(request: ScanRequest):
    """اسکن هوشمند بازار"""
    try:
        logger.info(f"🎯 دریافت درخواست اسکن: {request.symbols}")
        
        # اعتبارسنجی نمادها
        if not request.symbols:
            raise HTTPException(status_code=400, detail="لیست نمادها خالی است")
        
        # اجرای اسکن
        scan_result = await scan_engine.scan_symbols(request.symbols, request.scan_mode)
        
        # بررسی نتایج
        successful_scans = scan_result['summary']['successful']
        
        if successful_scans == 0:
            logger.warning("⚠️ هیچ اسکن موفقی انجام نشد")
        
        return {
            "status": "success",
            "message": f"اسکن کامل شد - {successful_scans} موفق از {len(request.symbols)}",
            "scan_mode": request.scan_mode,
            **scan_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_details = ErrorHandler.log_error(
            "AI Scan endpoint", 
            e,
            {'request_data': request.dict()}
        )
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "خطای داخلی در اسکن",
                "message": str(e),
                "debug_id": error_details.get('timestamp')
            }
        )

@api_router.get("/debug/coinstats")
async def debug_coinstats(symbol: str = "bitcoin"):
    """دیباگ مستقیم CoinStats API"""
    try:
        result = coin_stats.get_coin_data(symbol)
        return {
            "debug_mode": True,
            "symbol": symbol,
            "coinstats_status": coin_stats.get_status(),
            "api_response": result
        }
    except Exception as e:
        error_details = ErrorHandler.log_error("Debug coinstats", e)
        raise HTTPException(status_code=500, detail={
            "error": "خطا در دیباگ",
            "details": error_details
        })

# ثبت روت‌ها
app.include_router(api_router)

# ==================== مدیریت عمومی ====================

@app.get("/")
async def serve_frontend():
    """سرویس دهی فرانت‌اند"""
    try:
        return FileResponse("frontend/index.html")
    except Exception as e:
        logger.error(f"خطا در بارگذاری فرانت‌اند: {e}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "فایل فرانت‌اند یافت نشد",
                "detail": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/{path:path}")
async def catch_all(path: str):
    """مدیریت تمام مسیرهای دیگر"""
    if path.startswith('api/'):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Endpoint not found",
                "path": path,
                "available_endpoints": [
                    "/api/health",
                    "/api/system/status", 
                    "/api/ai/scan",
                    "/api/debug/coinstats"
                ],
                "timestamp": datetime.now().isoformat()
            }
        )
    else:
        try:
            return FileResponse("frontend/index.html")
        except:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Page not found",
                    "path": path,
                    "timestamp": datetime.now().isoformat()
                }
            )

# هندلر خطاهای全局
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    error_details = ErrorHandler.log_error(
        "Global 500 error", 
        exc,
        {'path': str(request.url), 'method': request.method}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "خطای داخلی سرور",
            "debug_id": error_details.get('timestamp'),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")
