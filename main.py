# main.py - با پردازش داده‌های خام CoinStats
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import traceback

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO)
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

# ==================== پردازش داده‌های خام CoinStats ====================

class CoinStatsDataProcessor:
    """پردازش داده‌های خام CoinStats"""
    
    @staticmethod
    def process_coin_list(raw_data: Dict) -> List[Dict]:
        """پردازش لیست کوین‌های خام"""
        try:
            if not raw_data or 'result' not in raw_data:
                return []
            
            coins = raw_data['result']
            processed_coins = []
            
            for coin in coins:
                processed_coin = {
                    'id': coin.get('id', ''),
                    'symbol': coin.get('symbol', ''),
                    'name': coin.get('name', ''),
                    'price': coin.get('price', 0),
                    'price_change_24h': coin.get('priceChange1d', 0),
                    'price_change_percent_24h': coin.get('priceChange1d', 0),  # ممکنه فیلد جداگانه داشته باشه
                    'volume_24h': coin.get('volume', 0),
                    'market_cap': coin.get('marketCap', 0),
                    'rank': coin.get('rank', 0),
                    'high_24h': coin.get('high', 0),
                    'low_24h': coin.get('low', 0),
                    'website': coin.get('websiteUrl', ''),
                    'raw_data': coin  # نگه‌داری داده خام
                }
                processed_coins.append(processed_coin)
            
            logger.info(f"✅ پردازش {len(processed_coins)} کوین انجام شد")
            return processed_coins
            
        except Exception as e:
            logger.error(f"❌ خطا در پردازش لیست کوین‌ها: {e}")
            return []
    
    @staticmethod
    def process_coin_details(raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """پردازش جزئیات کوین خام"""
        try:
            if not raw_data or 'result' not in raw_data:
                return {
                    'success': False,
                    'error': 'داده‌ای دریافت نشد',
                    'symbol': symbol
                }
            
            coin_data = raw_data['result']
            
            # استخراج فیلدهای مهم - با توجه به ساختار واقعی داده‌ها
            processed_data = {
                'success': True,
                'symbol': symbol,
                'id': coin_data.get('id', ''),
                'name': coin_data.get('name', ''),
                'price': float(coin_data.get('price', 0)),
                'price_change_24h': float(coin_data.get('priceChange1d', 0)),
                'price_change_percent_24h': float(coin_data.get('priceChange1d', 0)),
                'volume_24h': float(coin_data.get('volume', 0)),
                'market_cap': float(coin_data.get('marketCap', 0)),
                'rank': coin_data.get('rank', 0),
                'high_24h': float(coin_data.get('high', 0)),
                'low_24h': float(coin_data.get('low', 0)),
                'website': coin_data.get('websiteUrl', ''),
                'explorers': coin_data.get('explorers', []),
                'social_media': {
                    'twitter': coin_data.get('twitterUrl', ''),
                    'reddit': coin_data.get('redditUrl', '')
                },
                'timestamp': datetime.now().isoformat(),
                'raw_data_structure': list(coin_data.keys())  # برای دیباگ
            }
            
            logger.info(f"✅ پردازش جزئیات {symbol} انجام شد: ${processed_data['price']}")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ خطا در پردازش جزئیات {symbol}: {e}")
            return {
                'success': False,
                'error': f'خطا در پردازش داده: {str(e)}',
                'symbol': symbol,
                'raw_data': raw_data  # برای دیباگ
            }
    
    @staticmethod
    def process_chart_data(raw_data: Dict, symbol: str) -> Dict[str, Any]:
        """پردازش داده‌های چارت خام"""
        try:
            if not raw_data or 'result' not in raw_data:
                return {'success': False, 'error': 'داده چارت دریافت نشد'}
            
            chart_points = raw_data['result']
            processed_chart = {
                'success': True,
                'symbol': symbol,
                'data_points': len(chart_points),
                'prices': [point.get('price', 0) for point in chart_points],
                'timestamps': [point.get('timestamp', '') for point in chart_points],
                'sample_data': chart_points[:3] if chart_points else []  # نمونه‌ای از داده
            }
            
            return processed_chart
            
        except Exception as e:
            logger.error(f"❌ خطا در پردازش چارت {symbol}: {e}")
            return {'success': False, 'error': str(e)}

# ==================== مدیر CoinStats ====================

class CoinStatsManager:
    """مدیریت اتصال و پردازش داده‌های CoinStats"""
    
    def __init__(self):
        self.processor = CoinStatsDataProcessor()
        self.coin_stats_manager = None
        self.initialized = False
        
        self._initialize()
    
    def _initialize(self):
        """راه‌اندازی"""
        try:
            from complete_coinstats_manager import coin_stats_manager
            self.coin_stats_manager = coin_stats_manager
            self.initialized = True
            
            # تست دریافت داده
            test_data = self.coin_stats_manager.get_coins_list(limit=1)
            if test_data and 'result' in test_data and test_data['result']:
                logger.info("✅ CoinStats API قابل دسترسی است")
            else:
                logger.warning("⚠️ CoinStats API داده برنگرداند")
                
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی CoinStats: {e}")
            self.initialized = False
    
    def get_coin_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت و پردازش داده‌های کوین"""
        if not self.initialized:
            return {
                'success': False,
                'error': 'CoinStats Manager راه‌اندازی نشده',
                'symbol': symbol
            }
        
        try:
            logger.info(f"🔍 دریافت داده‌های {symbol}...")
            
            # دریافت داده خام
            raw_details = self.coin_stats_manager.get_coin_details(symbol, "USD")
            
            # پردازش داده خام
            processed_data = self.processor.process_coin_details(raw_details, symbol)
            
            if processed_data['success']:
                # دریافت داده چارت برای تحلیل بیشتر
                raw_charts = self.coin_stats_manager.get_coin_charts(symbol, "1w")
                chart_data = self.processor.process_chart_data(raw_charts, symbol)
                
                processed_data['chart_info'] = chart_data
                processed_data['data_quality'] = 'good' if processed_data['price'] > 0 else 'poor'
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های {symbol}: {e}")
            return {
                'success': False,
                'error': f'خطا در دریافت داده: {str(e)}',
                'symbol': symbol
            }
    
    def get_available_coins(self, limit: int = 50) -> List[Dict]:
        """دریافت لیست کوین‌های available"""
        try:
            raw_data = self.coin_stats_manager.get_coins_list(limit=limit)
            return self.processor.process_coin_list(raw_data)
        except Exception as e:
            logger.error(f"❌ خطا در دریافت لیست کوین‌ها: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """وضعیت سیستم"""
        return {
            'initialized': self.initialized,
            'timestamp': datetime.now().isoformat(),
            'available_coins_count': len(self.get_available_coins(10))
        }

# ایجاد مدیر
coin_stats_manager = CoinStatsManager()

# ==================== موتور اسکن ====================

class ScanEngine:
    """موتور اسکن با پردازش داده‌های واقعی"""
    
    def __init__(self):
        self.scan_count = 0
    
    def scan_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """اسکن نمادها"""
        self.scan_count += 1
        logger.info(f"🎯 شروع اسکن برای {len(symbols)} نماد")
        
        results = []
        successful = 0
        
        for symbol in symbols:
            try:
                # دریافت و پردازش داده
                coin_data = coin_stats_manager.get_coin_data(symbol)
                
                if coin_data['success']:
                    # تحلیل داده‌های پردازش شده
                    analysis = self._analyze_coin(coin_data)
                    results.append(analysis)
                    successful += 1
                    logger.info(f"✅ اسکن موفق {symbol}: ${analysis['price']}")
                else:
                    # خطا در دریافت داده
                    error_result = {
                        'symbol': symbol,
                        'success': False,
                        'error': coin_data.get('error', 'خطای ناشناخته'),
                        'price': 0,
                        'change_24h': 0,
                        'volume': 'N/A',
                        'market_cap': 'N/A',
                        'signal': 'ERROR',
                        'confidence': 0
                    }
                    results.append(error_result)
                    logger.warning(f"⚠️ اسکن ناموفق {symbol}: {coin_data.get('error')}")
                    
            except Exception as e:
                logger.error(f"❌ خطا در اسکن {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e),
                    'price': 0,
                    'change_24h': 0,
                    'volume': 'N/A',
                    'market_cap': 'N/A',
                    'signal': 'ERROR',
                    'confidence': 0
                })
        
        return {
            'scan_results': results,
            'summary': {
                'total': len(symbols),
                'successful': successful,
                'failed': len(symbols) - successful,
                'success_rate': f"{(successful/len(symbols))*100:.1f}%",
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _analyze_coin(self, coin_data: Dict) -> Dict[str, Any]:
        """تحلیل کوین پردازش شده"""
        price = coin_data.get('price', 0)
        change_24h = coin_data.get('price_change_24h', 0)
        
        # منطق ساده تحلیل
        if change_24h > 3:
            signal = "BUY"
            confidence = 0.7 + min(0.3, change_24h / 20)
        elif change_24h < -3:
            signal = "SELL"
            confidence = 0.6 + min(0.3, abs(change_24h) / 20)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            'symbol': coin_data['symbol'],
            'success': True,
            'price': price,
            'change_24h': change_24h,
            'volume': f"{coin_data.get('volume_24h', 0):,.0f}",
            'market_cap': f"{coin_data.get('market_cap', 0):,.0f}",
            'signal': signal,
            'confidence': round(confidence, 2),
            'timestamp': coin_data.get('timestamp'),
            'data_quality': coin_data.get('data_quality', 'unknown')
        }

# ایجاد موتور اسکن
scan_engine = ScanEngine()

# ==================== روت‌های API ====================

api_router = APIRouter(prefix="/api")

@api_router.get("/health")
async def health_check():
    """سلامت سیستم"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "coinstats_status": coin_stats_manager.get_status(),
        "total_scans": scan_engine.scan_count
    }

@api_router.post("/ai/scan")
async def ai_scan(request: ScanRequest):
    """اسکن هوشمند"""
    try:
        results = scan_engine.scan_symbols(request.symbols)
        
        return {
            "status": "success",
            "scan_mode": request.scan_mode,
            "real_data": True,
            **results
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در اسکن: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/debug/structure")
async def debug_structure(symbol: str = "bitcoin"):
    """دیباگ ساختار داده‌ها"""
    try:
        from complete_coinstats_manager import coin_stats_manager
        
        # دریافت داده خام
        raw_data = coin_stats_manager.get_coin_details(symbol, "USD")
        
        return {
            "symbol": symbol,
            "raw_structure": list(raw_data.keys()) if raw_data else "NO_DATA",
            "result_structure": list(raw_data['result'].keys()) if raw_data and 'result' in raw_data else "NO_RESULT",
            "sample_data": {k: raw_data['result'][k] for k in list(raw_data['result'].keys())[:10]} if raw_data and 'result' in raw_data else "NO_SAMPLE",
            "processed_data": coin_stats_manager.get_coin_data(symbol)
        }
    except Exception as e:
        return {"error": str(e)}

# ثبت روت‌ها
app.include_router(api_router)

# روت‌های عمومی
@app.get("/")
async def root():
    return {"message": "CryptoAI Scan API", "status": "running"}

@app.get("/{path:path}")
async def catch_all(path: str):
    if path.startswith('api/'):
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
