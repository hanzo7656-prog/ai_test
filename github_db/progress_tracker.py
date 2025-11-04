"""
ردیابی پیشرفت اسکن 500 ارزی
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    """ردیاب پیشرفت اسکن"""
    
    def __init__(self, cache_dir: str = "./github_db_data"):
        self.cache_dir = Path(cache_dir)
        self.progress_file = self.cache_dir / "metadata" / "scan_progress.json"
        
    def update_progress(self, total_symbols: int, scanned: int, 
                       current_batch: int, status: str = "running") -> bool:
        """بروزرسانی پیشرفت"""
        try:
            progress_data = {
                "total_symbols": total_symbols,
                "scanned": scanned,
                "current_batch": current_batch,
                "percent_complete": round((scanned / total_symbols) * 100, 2),
                "status": status,
                "last_updated": datetime.now().isoformat(),
                "estimated_completion": self._estimate_completion(scanned, total_symbols)
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📈 پیشرفت بروز شد: {scanned}/{total_symbols} ({progress_data['percent_complete']}%)")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در بروزرسانی پیشرفت: {e}")
            return False
    
    def get_progress(self) -> Dict[str, Any]:
        """دریافت پیشرفت فعلی"""
        try:
            if not self.progress_file.exists():
                return {
                    "total_symbols": 0,
                    "scanned": 0,
                    "current_batch": 0,
                    "percent_complete": 0,
                    "status": "not_started",
                    "last_updated": datetime.now().isoformat()
                }
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"❌ خطا در خواندن پیشرفت: {e}")
            return {
                "total_symbols": 0,
                "scanned": 0,
                "current_batch": 0,
                "percent_complete": 0,
                "status": "error",
                "last_updated": datetime.now().isoformat()
            }
    
    def _estimate_completion(self, scanned: int, total: int) -> str:
        """تخمین زمان تکمیل"""
        if scanned == 0:
            return "Unknown"
        
        time_per_symbol = 2  # ثانیه به ازای هر ارز (تخمین)
        remaining_symbols = total - scanned
        remaining_seconds = remaining_symbols * time_per_symbol
        
        if remaining_seconds < 60:
            return f"{int(remaining_seconds)} ثانیه"
        elif remaining_seconds < 3600:
            return f"{int(remaining_seconds / 60)} دقیقه"
        else:
            return f"{int(remaining_seconds / 3600)} ساعت"
    
    def reset_progress(self) -> bool:
        """بازنشانی پیشرفت"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            logger.info("🔄 پیشرفت بازنشانی شد")
            return True
        except Exception as e:
            logger.error(f"❌ خطا در بازنشانی پیشرفت: {e}")
            return False
