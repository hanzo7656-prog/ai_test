# 📁 src/data/minimal_processors/stream_processor.py

import asyncio
from typing import Dict, List, Any, Callable, Optional
from collections import deque
from ...utils.memory_monitor import MemoryMonitor

class StreamProcessor:
    """پردازشگر جریانی برای داده‌های بلادرنگ بازار"""
    
    def __init__(self, window_size: int = 100, max_memory_mb: int = 50):
        self.window_size = window_size  # اندازه پنجره لغزان
        self.max_memory_mb = max_memory_mb
        self.data_window = deque(maxlen=window_size)
        self.memory_monitor = MemoryMonitor()
        self.processors = []
        
    def add_processor(self, processor: Callable):
        """افزودن پردازشگر به پipeline"""
        self.processors.append(processor)
    
    async def process_stream(self, data_stream: Any) -> List[Dict]:
        """پردازش جریانی داده‌های ورودی"""
        processed_data = []
        
        async for data_chunk in data_stream:
            if self._memory_exceeded():
                self._cleanup_old_data()
            
            # اعمال تمام پردازشگرها
            processed_chunk = data_chunk
            for processor in self.processors:
                processed_chunk = processor(processed_chunk)
            
            # افزودن به پنجره داده
            self.data_window.append(processed_chunk)
            processed_data.append(processed_chunk)
            
            self.memory_monitor.log_memory_usage("stream_processing")
        
        return processed_data
    
    def _memory_exceeded(self) -> bool:
        """بررسی превыن حافظه مجاز"""
        current_usage = self.memory_monitor.get_current_usage()
        return current_usage > self.max_memory_mb * 1024 * 1024  # تبدیل به بایت
    
    def _cleanup_old_data(self):
        """پاک‌سازی داده‌های قدیمی"""
        if len(self.data_window) > self.window_size // 2:
            # حفظ نیمی از داده‌های اخیر
            keep_count = self.window_size // 2
            self.data_window = deque(
                list(self.data_window)[-keep_count:], 
                maxlen=self.window_size
            )
    
    def get_window_stats(self) -> Dict:
        """آمار پنجره داده جاری"""
        return {
            "window_size": len(self.data_window),
            "max_window_size": self.window_size,
            "memory_usage_mb": self.memory_monitor.get_current_usage() / (1024 * 1024),
            "max_memory_mb": self.max_memory_mb
        }
