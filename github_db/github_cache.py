"""
سیستم کش پیشرفته مبتنی بر GitHub برای VortexAI
مدیریت ذخیره‌سازی، بازیابی و فشرده‌سازی داده‌ها
"""

import os
import json
import gzip
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class GitHubDBCache:
    """مدیریت کش پیشرفته با GitHub"""
    
    def __init__(self, repo_path: str = "./github_db_data"):
        self.repo_path = Path(repo_path)
        self.setup_directories()
        
        # تنظیمات فشرده‌سازی
        self.compression_threshold_days = 7  # بعد از 7 روز فشرده شود
        self.cleanup_threshold_days = 30     # بعد از 30 روز پاک شود
        
    def setup_directories(self):
        """ایجاد ساختار دایرکتوری‌ها"""
        directories = [
            "live_data",
            "compressed_data", 
            "metadata",
            "batch_progress",
            "symbols_list"
        ]
        
        for dir_name in directories:
            dir_path = self.repo_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 دایرکتوری ایجاد شد: {dir_path}")
    
    def save_live_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """ذخیره داده‌های زنده"""
        try:
            symbol_file = self.repo_path / "live_data" / f"{symbol.lower()}.json"
            
            # اضافه کردن متادیتا
            enriched_data = {
                "symbol": symbol,
                "last_updated": datetime.now().isoformat(),
                "data": data,
                "version": "1.0"
            }
            
            with open(symbol_file, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 داده زنده ذخیره شد: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره داده {symbol}: {e}")
            return False
    
    def get_live_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """دریافت داده‌های زنده"""
        try:
            symbol_file = self.repo_path / "live_data" / f"{symbol.lower()}.json"
            
            if not symbol_file.exists():
                return None
            
            with open(symbol_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # بررسی تاریخ انقضا
            last_updated = datetime.fromisoformat(data['last_updated'])
            if datetime.now() - last_updated > timedelta(minutes=10):
                logger.warning(f"⚠️ داده {symbol} قدیمی است")
                return None
            
            return data['data']
            
        except Exception as e:
            logger.error(f"❌ خطا در خواندن داده {symbol}: {e}")
            return None
    
    def compress_old_data(self, symbol: str) -> bool:
        """فشرده‌سازی داده‌های قدیمی"""
        try:
            symbol_file = self.repo_path / "live_data" / f"{symbol.lower()}.json"
            compressed_dir = self.repo_path / "compressed_data"
            
            if not symbol_file.exists():
                return False
            
            # خواندن داده فعلی
            with open(symbol_file, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            
            last_updated = datetime.fromisoformat(current_data['last_updated'])
            
            # بررسی是否需要 فشرده‌سازی
            if datetime.now() - last_updated < timedelta(days=self.compression_threshold_days):
                return False
            
            # ایجاد فایل فشرده
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            compressed_file = compressed_dir / f"{symbol.lower()}_{timestamp}.json.gz"
            
            with gzip.open(compressed_file, 'wt', encoding='utf-8') as f:
                json.dump(current_data, f, ensure_ascii=False)
            
            # حذف فایل اصلی
            symbol_file.unlink()
            
            logger.info(f"📦 داده {symbol} فشرده شد: {compressed_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در فشرده‌سازی {symbol}: {e}")
            return False
    
    def save_batch_progress(self, batch_id: str, progress: Dict[str, Any]) -> bool:
        """ذخیره پیشرفت اسکن دسته‌ای"""
        try:
            progress_file = self.repo_path / "batch_progress" / f"{batch_id}.json"
            
            progress_data = {
                "batch_id": batch_id,
                "last_updated": datetime.now().isoformat(),
                "progress": progress
            }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره پیشرفت {batch_id}: {e}")
            return False
    
    def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """دریافت پیشرفت اسکن دسته‌ای"""
        try:
            progress_file = self.repo_path / "batch_progress" / f"{batch_id}.json"
            
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"❌ خطا در خواندن پیشرفت {batch_id}: {e}")
            return None
    
    def save_symbols_list(self, symbols: List[str], list_name: str = "top_500") -> bool:
        """ذخیره لیست ارزها"""
        try:
            symbols_file = self.repo_path / "symbols_list" / f"{list_name}.json"
            
            symbols_data = {
                "name": list_name,
                "count": len(symbols),
                "last_updated": datetime.now().isoformat(),
                "symbols": symbols
            }
            
            with open(symbols_file, 'w', encoding='utf-8') as f:
                json.dump(symbols_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 لیست ارزها ذخیره شد: {list_name} ({len(symbols)} ارز)")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره لیست ارزها: {e}")
            return False
    
    def get_symbols_list(self, list_name: str = "top_500") -> Optional[List[str]]:
        """دریافت لیست ارزها"""
        try:
            symbols_file = self.repo_path / "symbols_list" / f"{list_name}.json"
            
            if not symbols_file.exists():
                return None
            
            with open(symbols_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('symbols', [])
                
        except Exception as e:
            logger.error(f"❌ خطا در خواندن لیست ارزها: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """آمار کش"""
        try:
            live_data_dir = self.repo_path / "live_data"
            compressed_dir = self.repo_path / "compressed_data"
            
            live_files = list(live_data_dir.glob("*.json"))
            compressed_files = list(compressed_dir.glob("*.gz"))
            
            # محاسبه حجم
            live_size = sum(f.stat().st_size for f in live_files)
            compressed_size = sum(f.stat().st_size for f in compressed_files)
            
            return {
                "live_files_count": len(live_files),
                "live_size_mb": round(live_size / (1024 * 1024), 2),
                "compressed_files_count": len(compressed_files),
                "compressed_size_mb": round(compressed_size / (1024 * 1024), 2),
                "total_size_mb": round((live_size + compressed_size) / (1024 * 1024), 2),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در محاسبه آمار کش: {e}")
            return {}
    
    def cleanup_old_data(self) -> int:
        """پاکسازی داده‌های بسیار قدیمی"""
        try:
            compressed_dir = self.repo_path / "compressed_data"
            deleted_count = 0
            
            for compressed_file in compressed_dir.glob("*.gz"):
                file_time = datetime.fromtimestamp(compressed_file.stat().st_mtime)
                
                if datetime.now() - file_time > timedelta(days=self.cleanup_threshold_days):
                    compressed_file.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ فایل قدیمی حذف شد: {compressed_file.name}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ خطا در پاکسازی داده‌های قدیمی: {e}")
            return 0
