"""
مدیریت فشرده‌سازی داده‌های تاریخی
"""

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataCompressor:
    """مدیریت فشرده‌سازی هوشمند"""
    
    def __init__(self, cache_dir: str = "./github_db_data"):
        self.cache_dir = Path(cache_dir)
        
    def compress_old_files(self, days_threshold: int = 7) -> int:
        """فشرده‌سازی فایل‌های قدیمی"""
        compressed_count = 0
        live_data_dir = self.cache_dir / "live_data"
        
        if not live_data_dir.exists():
            return 0
        
        for json_file in live_data_dir.glob("*.json"):
            try:
                # بررسی تاریخ فایل
                file_time = datetime.fromtimestamp(json_file.stat().st_mtime)
                if datetime.now() - file_time > timedelta(days=days_threshold):
                    
                    # خواندن و فشرده‌سازی
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # ایجاد فایل فشرده
                    compressed_file = self.cache_dir / "compressed_data" / f"{json_file.stem}.json.gz"
                    
                    with gzip.open(compressed_file, 'wt', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                    
                    # حذف فایل اصلی
                    json_file.unlink()
                    compressed_count += 1
                    
                    logger.info(f"📦 فایل فشرده شد: {json_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ خطا در فشرده‌سازی {json_file}: {e}")
        
        return compressed_count
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """آمار فشرده‌سازی"""
        compressed_dir = self.cache_dir / "compressed_data"
        
        if not compressed_dir.exists():
            return {"compressed_files": 0, "total_size_mb": 0}
        
        compressed_files = list(compressed_dir.glob("*.gz"))
        total_size = sum(f.stat().st_size for f in compressed_files)
        
        return {
            "compressed_files": len(compressed_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "last_compression": datetime.now().isoformat()
        }
