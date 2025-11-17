import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)

class MemoryManager:
    """مدیریت حافظه ۳ لایه‌ای برای هوش مصنوعی"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_manager = None
        
        # لایه‌های حافظه
        self.sensory_memory = {}      # حافظه حسی (کوتاه‌مدت)
        self.working_memory = {}      # حافظه فعال (میان‌مدت)  
        self.long_term_memory = {}    # حافظه بلندمدت (دائمی)
        
        # تنظیمات TTL
        self.sensory_ttl = config.get('sensory_ttl_hours', 24) * 3600
        self.working_ttl = config.get('working_ttl_days', 30) * 24 * 3600
        
        # آستانه‌های انتقال بین لایه‌ها
        self.access_threshold = 3     # تعداد دسترسی برای انتقال به لایه بالاتر
        self.importance_threshold = 0.7  # آستانه اهمیت برای انتقال
        
        # آمار استفاده
        self.access_stats = defaultdict(int)
        self.creation_times = {}
        
        logger.info("🚀 مدیر حافظه ۳ لایه‌ای راه‌اندازی شد")
    
    def initialize_redis(self, redis_manager):
        """مقداردهی اولیه اتصال ردیس"""
        self.redis_manager = redis_manager
        logger.info("✅ اتصال ردیس برای مدیر حافظه تنظیم شد")
    
    def store_sensory(self, key: str, data: Any, user_id: str = "default"):
        """ذخیره در حافظه حسی"""
        sensory_key = f"sensory:{user_id}:{key}"
        
        memory_item = {
            'data': data,
            'timestamp': time.time(),
            'access_count': 0,
            'importance': 0.1,  # اهمیت اولیه پایین
            'user_id': user_id,
            'type': 'sensory'
        }
        
        self.sensory_memory[sensory_key] = memory_item
        self.creation_times[sensory_key] = time.time()
        
        logger.debug(f"🧠 ذخیره در حافظه حسی: {sensory_key}")
    
    def store_working(self, key: str, data: Any, user_id: str = "default"):
        """ذخیره در حافظه فعال"""
        working_key = f"working:{user_id}:{key}"
        
        memory_item = {
            'data': data,
            'timestamp': time.time(),
            'access_count': 0,
            'importance': 0.5,  # اهمیت متوسط
            'user_id': user_id,
            'type': 'working'
        }
        
        self.working_memory[working_key] = memory_item
        self.creation_times[working_key] = time.time()
        
        # ذخیره در ردیس اگر متصل باشد
        if self.redis_manager:
            success, _ = self.redis_manager.set(
                "mother_a", working_key, memory_item, self.working_ttl
            )
            if success:
                logger.debug(f"💾 ذخیره در حافظه فعال (ردیس): {working_key}")
        
        logger.debug(f"🧠 ذخیره در حافظه فعال: {working_key}")
    
    def store_long_term(self, key: str, data: Any, user_id: str = "default"):
        """ذخیره در حافظه بلندمدت"""
        long_term_key = f"long_term:{user_id}:{key}"
        
        # فشرده‌سازی داده برای صرفه‌جویی در فضای
        compressed_data = self._compress_data(data)
        
        memory_item = {
            'data': compressed_data,
            'timestamp': time.time(),
            'access_count': 0,
            'importance': 0.9,  # اهمیت بالا
            'user_id': user_id,
            'type': 'long_term',
            'compressed': True
        }
        
        self.long_term_memory[long_term_key] = memory_item
        self.creation_times[long_term_key] = time.time()
        
        # ذخیره دائمی در ردیس
        if self.redis_manager:
            success, _ = self.redis_manager.set(
                "mother_a", long_term_key, memory_item, 365 * 24 * 3600  # 1 سال
            )
            if success:
                logger.info(f"💾 ذخیره دائمی در حافظه بلندمدت: {long_term_key}")
    
    def retrieve(self, key: str, user_id: str = "default") -> Optional[Any]:
        """بازیابی داده از حافظه (جستجوی سلسله‌مراتبی)"""
        # جستجو در حافظه حسی
        sensory_key = f"sensory:{user_id}:{key}"
        if sensory_key in self.sensory_memory:
            item = self._access_memory_item(sensory_key, 'sensory')
            self._consider_promotion(sensory_key, item)
            return item['data']
        
        # جستجو در حافظه فعال
        working_key = f"working:{user_id}:{key}"
        if working_key in self.working_memory:
            item = self._access_memory_item(working_key, 'working')
            self._consider_promotion(working_key, item)
            return item['data']
        
        # جستجو در حافظه بلندمدت
        long_term_key = f"long_term:{user_id}:{key}"
        if long_term_key in self.long_term_memory:
            item = self._access_memory_item(long_term_key, 'long_term')
            return item['data']
        
        # جستجو در ردیس برای حافظه‌های پایدار
        if self.redis_manager:
            # جستجو در حافظه فعال ردیس
            working_data, _ = self.redis_manager.get("mother_a", working_key)
            if working_data:
                self.working_memory[working_key] = working_data
                item = self._access_memory_item(working_key, 'working')
                return item['data']
            
            # جستجو در حافظه بلندمدت ردیس
            long_term_data, _ = self.redis_manager.get("mother_a", long_term_key)
            if long_term_data:
                self.long_term_memory[long_term_key] = long_term_data
                item = self._access_memory_item(long_term_key, 'long_term')
                return item['data']
        
        logger.debug(f"🔍 داده یافت نشد: {key}")
        return None
    
    def _access_memory_item(self, key: str, memory_type: str) -> Dict[str, Any]:
        """ثبت دسترسی به آیتم حافظه و به‌روزرسانی آمار"""
        if memory_type == 'sensory':
            item = self.sensory_memory[key]
        elif memory_type == 'working':
            item = self.working_memory[key]
        else:
            item = self.long_term_memory[key]
        
        # به‌روزرسانی آمار دسترسی
        item['access_count'] += 1
        item['last_accessed'] = time.time()
        self.access_stats[key] += 1
        
        # افزایش اهمیت بر اساس دسترسی
        item['importance'] = min(1.0, item['importance'] + 0.05)
        
        return item
    
    def _consider_promotion(self, key: str, item: Dict[str, Any]):
        """بررسی ارتقاء آیتم به لایه بالاتر حافظه"""
        current_time = time.time()
        age = current_time - item['timestamp']
        
        # شرایط ارتقاء به حافظه فعال
        if (item['type'] == 'sensory' and 
            item['access_count'] >= self.access_threshold and 
            age > 3600):  # حداقل 1 ساعت
            
            self._promote_to_working(key, item)
        
        # شرایط ارتقاء به حافظه بلندمدت
        elif (item['type'] == 'working' and 
              item['importance'] >= self.importance_threshold and 
              age > (7 * 24 * 3600)):  # حداقل 1 هفته
            
            self._promote_to_long_term(key, item)
    
    def _promote_to_working(self, sensory_key: str, item: Dict[str, Any]):
        """ارتقاء از حافظه حسی به فعال"""
        working_key = sensory_key.replace('sensory:', 'working:')
        
        # انتقال داده
        self.store_working(working_key.split(':')[-1], item['data'], item['user_id'])
        
        # حذف از حافظه حسی
        del self.sensory_memory[sensory_key]
        
        logger.info(f"🔼 ارتقاء از حافظه حسی به فعال: {sensory_key} → {working_key}")
    
    def _promote_to_long_term(self, working_key: str, item: Dict[str, Any]):
        """ارتقاء از حافظه فعال به بلندمدت"""
        long_term_key = working_key.replace('working:', 'long_term:')
        
        # انتقال داده
        self.store_long_term(long_term_key.split(':')[-1], item['data'], item['user_id'])
        
        # حذف از حافظه فعال
        del self.working_memory[working_key]
        
        logger.info(f"🔼 ارتقاء از حافظه فعال به بلندمدت: {working_key} → {long_term_key}")
    
    def _compress_data(self, data: Any) -> Any:
        """فشرده‌سازی داده برای ذخیره‌سازی بهینه"""
        if isinstance(data, str) and len(data) > 100:
            # فشرده‌سازی متن‌های طولانی
            return data[:100] + "..." if len(data) > 100 else data
        
        elif isinstance(data, dict):
            # فشرده‌سازی دیکشنری - حفظ کلیدهای مهم
            compressed = {}
            important_keys = ['type', 'intent', 'concept', 'pattern', 'essential_data']
            
            for key, value in data.items():
                if key in important_keys or len(str(value)) < 50:
                    compressed[key] = value
            
            return compressed if compressed else data
        
        elif isinstance(data, list) and len(data) > 10:
            # فشرده‌سازی لیست‌های بزرگ
            return data[:10] + [f"...({len(data)-10} موارد دیگر)"]
        
        return data
    
    def cleanup_expired(self):
        """پاک‌سازی داده‌های منقضی شده"""
        current_time = time.time()
        cleaned_count = 0
        
        # پاک‌سازی حافظه حسی
        expired_sensory = [
            key for key, item in self.sensory_memory.items()
            if current_time - item['timestamp'] > self.sensory_ttl
        ]
        
        for key in expired_sensory:
            del self.sensory_memory[key]
            cleaned_count += 1
        
        # پاک‌سازی حافظه فعال
        expired_working = [
            key for key, item in self.working_memory.items()
            if current_time - item['timestamp'] > self.working_ttl
        ]
        
        for key in expired_working:
            del self.working_memory[key]
            cleaned_count += 1
        
        logger.info(f"🧹 پاک‌سازی {cleaned_count} آیتم منقضی شده")
        return cleaned_count
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """آمار وضعیت حافظه"""
        current_time = time.time()
        
        return {
            'sensory_memory': {
                'count': len(self.sensory_memory),
                'oldest_item_seconds': self._get_oldest_age(self.sensory_memory, current_time),
                'total_accesses': sum(item['access_count'] for item in self.sensory_memory.values())
            },
            'working_memory': {
                'count': len(self.working_memory),
                'oldest_item_days': self._get_oldest_age(self.working_memory, current_time) / 86400,
                'total_accesses': sum(item['access_count'] for item in self.working_memory.values())
            },
            'long_term_memory': {
                'count': len(self.long_term_memory),
                'oldest_item_days': self._get_oldest_age(self.long_term_memory, current_time) / 86400,
                'total_accesses': sum(item['access_count'] for item in self.long_term_memory.values())
            },
            'total_memory_usage_mb': self._calculate_memory_usage()
        }
    
    def _get_oldest_age(self, memory_dict: Dict[str, Any], current_time: float) -> float:
        """محاسبه سن قدیمی‌ترین آیتم"""
        if not memory_dict:
            return 0
        oldest_timestamp = min(item['timestamp'] for item in memory_dict.values())
        return current_time - oldest_timestamp
    
    def _calculate_memory_usage(self) -> float:
        """محاسبه استفاده از حافظه"""
        total_size = 0
        
        for memory_dict in [self.sensory_memory, self.working_memory, self.long_term_memory]:
            for key, item in memory_dict.items():
                total_size += len(str(key).encode('utf-8'))
                total_size += len(str(item).encode('utf-8'))
        
        return round(total_size / (1024 * 1024), 2)  # به مگابایت
