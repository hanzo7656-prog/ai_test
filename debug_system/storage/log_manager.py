import logging
import json
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

class LogManager:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # بافر برای لاگ‌های Real-Time
        self.log_buffer = deque(maxlen=10000)
        self._buffer_lock = threading.Lock()
        
        # تنظیمات rotation
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.retention_days = 30
        
        # شروع background task برای نوشتن لاگ‌ها
        self._start_log_writer()
        
    def _start_log_writer(self):
        """شروع background task برای نوشتن لاگ‌ها"""
        def log_writer_loop():
            while True:
                try:
                    self._flush_buffer_to_disk()
                    threading.Event().wait(10)  # هر ۱۰ ثانیه
                except Exception as e:
                    logger.error(f"❌ Log writer error: {e}")
                    threading.Event().wait(30)
        
        writer_thread = threading.Thread(target=log_writer_loop, daemon=True)
        writer_thread.start()
        logger.info("✅ Log writer started")
    
    def log_endpoint_call(self, endpoint_data: Dict[str, Any]):
        """ثبت لاگ فراخوانی اندپوینت"""
        log_entry = {
            'type': 'endpoint_call',
            'timestamp': datetime.now().isoformat(),
            'data': endpoint_data
        }
        
        self._add_to_buffer(log_entry)
    
    def log_system_metrics(self, metrics_data: Dict[str, Any]):
        """ثبت لاگ متریک‌های سیستم"""
        log_entry = {
            'type': 'system_metrics',
            'timestamp': datetime.now().isoformat(), 
            'data': metrics_data
        }
        
        self._add_to_buffer(log_entry)
    
    def log_security_event(self, security_data: Dict[str, Any]):
        """ثبت لاگ رویداد امنیتی"""
        log_entry = {
            'type': 'security_event',
            'timestamp': datetime.now().isoformat(),
            'data': security_data
        }
        
        self._add_to_buffer(log_entry)
    
    def log_performance_alert(self, alert_data: Dict[str, Any]):
        """ثبت لاگ هشدار عملکرد"""
        log_entry = {
            'type': 'performance_alert',
            'timestamp': datetime.now().isoformat(),
            'data': alert_data
        }
        
        self._add_to_buffer(log_entry)
    
    def _add_to_buffer(self, log_entry: Dict[str, Any]):
        """اضافه کردن لاگ به بافر"""
        with self._buffer_lock:
            self.log_buffer.append(log_entry)
    
    def _flush_buffer_to_disk(self):
        """نوشتن بافر به دیسک"""
        if not self.log_buffer:
            return
            
        with self._buffer_lock:
            logs_to_write = list(self.log_buffer)
            self.log_buffer.clear()
        
        if not logs_to_write:
            return
        
        # گروه‌بندی لاگ‌ها بر اساس نوع و تاریخ
        grouped_logs = {}
        for log in logs_to_write:
            log_date = datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d')
            log_type = log['type']
            key = f"{log_date}_{log_type}"
            
            if key not in grouped_logs:
                grouped_logs[key] = []
            grouped_logs[key].append(log)
        
        # نوشتن هر گروه در فایل مربوطه
        for key, logs in grouped_logs.items():
            filename = self.log_dir / f"{key}.log"
            self._write_logs_to_file(filename, logs)
    
    def _write_logs_to_file(self, filename: Path, logs: List[Dict]):
        """نوشتن لاگ‌ها به فایل"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                for log in logs:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"❌ Error writing logs to {filename}: {e}")
    
    def get_logs(self, 
                 log_type: str = None,
                 start_date: datetime = None,
                 end_date: datetime = None,
                 limit: int = 1000) -> List[Dict[str, Any]]:
        """دریافت لاگ‌ها با فیلتر"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=1)
        if end_date is None:
            end_date = datetime.now()
        
        logs = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            if log_type:
                # جستجو برای نوع خاص
                filename = self.log_dir / f"{date_str}_{log_type}.log"
                if filename.exists():
                    logs.extend(self._read_log_file(filename, limit))
            else:
                # جستجو برای تمام انواع
                for file in self.log_dir.glob(f"{date_str}_*.log"):
                    logs.extend(self._read_log_file(file, limit))
            
            current_date += timedelta(days=1)
        
        # مرتب‌سازی بر اساس timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return logs[:limit]
    
    def _read_log_file(self, filename: Path, limit: int) -> List[Dict]:
        """خواندن لاگ‌ها از فایل"""
        logs = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            log_entry = json.loads(line.strip())
                            logs.append(log_entry)
                            if len(logs) >= limit:
                                break
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"❌ Error reading log file {filename}: {e}")
        
        return logs
    
    def compress_old_logs(self):
        """فشرده‌سازی لاگ‌های قدیمی"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in self.log_dir.glob("*.log"):
            file_date_str = log_file.stem.split('_')[0]
            try:
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                if file_date < cutoff_date:
                    self._compress_file(log_file)
            except ValueError:
                continue
    
    def _compress_file(self, file_path: Path):
        """فشرده‌سازی یک فایل"""
        try:
            compressed_path = file_path.with_suffix('.log.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # حذف فایل اصلی پس از فشرده‌سازی
            file_path.unlink()
            logger.info(f"✅ Compressed log file: {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error compressing {file_path}: {e}")
    
    def get_log_statistics(self, days: int = 7) -> Dict[str, Any]:
        """دریافت آمار لاگ‌ها"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stats = {
            'total_logs': 0,
            'by_type': defaultdict(int),
            'by_date': defaultdict(int),
            'largest_log_file': {'name': '', 'size_mb': 0}
        }
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            for log_file in self.log_dir.glob(f"{date_str}_*.log"):
                # اندازه فایل
                file_size_mb = log_file.stat().st_size / (1024 * 1024)
                
                if file_size_mb > stats['largest_log_file']['size_mb']:
                    stats['largest_log_file'] = {
                        'name': log_file.name,
                        'size_mb': round(file_size_mb, 2)
                    }
                
                # شمارش لاگ‌ها
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_count = sum(1 for _ in f)
                        stats['total_logs'] += log_count
                        
                        # تشخیص نوع از نام فایل
                        log_type = log_file.stem.split('_')[1]
                        stats['by_type'][log_type] += log_count
                        stats['by_date'][date_str] += log_count
                        
                except Exception as e:
                    logger.error(f"❌ Error reading {log_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return stats
    
    def cleanup_old_logs(self):
        """پاک‌سازی لاگ‌های قدیمی"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("*.log*"):  # شامل فایل‌های فشرده هم می‌شود
            file_date_str = log_file.stem.split('_')[0]
            try:
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"🧹 Deleted old log file: {log_file.name}")
            except ValueError:
                # اگر فرمت تاریخ درست نبود، فایل را حذف نکن
                continue

# ایجاد نمونه گلوبال
log_manager = LogManager()
