"""
اسکنر دسته‌ای برای اسکن 25 تایی ارزها
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from .github_cache import GitHubDBCache

logger = logging.getLogger(__name__)

class BatchScanner:
    """اسکنر دسته‌ای هوشمند"""
    
    def __init__(self, github_cache: GitHubDBCache, batch_size: int = 25):
        self.cache = github_cache
        self.batch_size = batch_size
        
    async def scan_batch(self, symbols: List[str], scan_type: str = "basic") -> Dict[str, Any]:
        """اسکن یک دسته از ارزها"""
        start_time = time.time()
        results = {}
        successful_scans = 0
        
        logger.info(f"🔍 شروع اسکن دسته‌ای: {len(symbols)} ارز - نوع: {scan_type}")
        
        for i, symbol in enumerate(symbols):
            try:
                # اسکن ارز
                if scan_type == "ai":
                    coin_data = await self._scan_ai(symbol)
                else:
                    coin_data = await self._scan_basic(symbol)
                
                if coin_data and "error" not in coin_data:
                    # ذخیره در کش
                    self.cache.save_live_data(symbol, coin_data)
                    successful_scans += 1
                    
                    results[symbol] = {
                        "status": "success",
                        "data": coin_data
                    }
                else:
                    results[symbol] = {
                        "status": "error",
                        "error": coin_data.get("error", "Unknown error") if coin_data else "Scan failed"
                    }
                
                # لاگ پیشرفت
                if (i + 1) % 5 == 0:
                    logger.info(f"📊 پیشرفت اسکن: {i + 1}/{len(symbols)}")
                    
            except Exception as e:
                logger.error(f"❌ خطا در اسکن {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }
        
        total_time = time.time() - start_time
        
        # ذخیره نتایج دسته
        batch_result = {
            "batch_size": len(symbols),
            "successful_scans": successful_scans,
            "failed_scans": len(symbols) - successful_scans,
            "total_time_seconds": round(total_time, 2),
            "average_time_per_symbol": round(total_time / len(symbols), 2),
            "scan_type": scan_type,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        logger.info(f"✅ اسکن دسته‌ای کامل: {successful_scans}/{len(symbols)} موفق")
        
        return batch_result
    
    async def _scan_ai(self, symbol: str) -> Dict[str, Any]:
        """اسکن AI - شبیه‌سازی"""
        # اینجا باید با main.py یکپارچه شود
        await asyncio.sleep(0.1)  # شبیه‌سازی تاخیر
        
        return {
            "symbol": symbol,
            "scan_type": "ai",
            "timestamp": datetime.now().isoformat(),
            "data": f"AI data for {symbol}"
        }
    
    async def _scan_basic(self, symbol: str) -> Dict[str, Any]:
        """اسکن معمولی - شبیه‌سازی"""
        # اینجا باید با main.py یکپارچه شود
        await asyncio.sleep(0.05)  # شبیه‌سازی تاخیر
        
        return {
            "symbol": symbol,
            "scan_type": "basic",
            "timestamp": datetime.now().isoformat(),
            "price": 1000.0,
            "change_24h": 2.5,
            "volume": 50000000
        }
    
    def get_scan_progress(self, total_symbols: int, current_batch: int, current_in_batch: int) -> Dict[str, Any]:
        """محاسبه پیشرفت کلی"""
        total_scanned = (current_batch * self.batch_size) + current_in_batch
        percent_complete = (total_scanned / total_symbols) * 100
        
        return {
            "total_symbols": total_symbols,
            "scanned_so_far": total_scanned,
            "percent_complete": round(percent_complete, 2),
            "current_batch": current_batch + 1,
            "current_in_batch": current_in_batch + 1,
            "batch_size": self.batch_size,
            "remaining_batches": (total_symbols - total_scanned) // self.batch_size,
            "timestamp": datetime.now().isoformat()
        }
