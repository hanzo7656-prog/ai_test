import redis
import json
import os
import time
from datetime import datetime
from typing import Any, Optional, Tuple, List, Dict

class RedisCacheManager:
    def __init__(self):
        self.databases = {
            'uta': None,      # هسته مدل AI - داده‌های حیاتی
            'utb': None,      # پردازش AI - داده‌های نیمه‌عمر  
            'utc': None,      # داده‌های خام - تاریخی + فشرده
            'mother_a': None, # پردازش سیستم - داده‌های حیاتی
            'mother_b': None  # عملیات و کش - داده‌های موقت
        }
        self._connect_all()
        
    def _connect_all(self):
        """اتصال به تمام دیتابیس‌های Redis از Environment Variables در Render"""
        try:
            # UTA_REDIS_AI - هسته مدل AI (داده‌های حیاتی)
            self.databases['uta'] = redis.Redis.from_url(
                os.getenv("UTA_REDIS_AI"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=5
            )
            self.databases['uta'].ping()
            print("✅ UTA_REDIS_AI connected successfully!")
            
            # UTB_REDIS_AI - پردازش AI (داده‌های نیمه‌عمر)
            self.databases['utb'] = redis.Redis.from_url(
                os.getenv("UTB_REDIS_AI"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=5
            )
            self.databases['utb'].ping()
            print("✅ UTB_REDIS_AI connected successfully!")
            
            # UTC_REDIS_AI - داده‌های خام (تاریخی + فشرده)
            self.databases['utc'] = redis.Redis.from_url(
                os.getenv("UTC_REDIS_AI"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=5
            )
            self.databases['utc'].ping()
            print("✅ UTC_REDIS_AI connected successfully!")
            
            # MOTHER_A_URL - پردازش سیستم (داده‌های حیاتی)
            self.databases['mother_a'] = redis.Redis.from_url(
                os.getenv("MOTHER_A_URL"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=5
            )
            self.databases['mother_a'].ping()
            print("✅ MOTHER_A_URL connected successfully!")
            
            # MOTHER_B_URL - عملیات و کش (داده‌های موقت)
            self.databases['mother_b'] = redis.Redis.from_url(
                os.getenv("MOTHER_B_URL"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=5
            )
            self.databases['mother_b'].ping()
            print("✅ MOTHER_B_URL connected successfully!")
            
            print("🎯 All 5 Redis databases connected and ready!")
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            # میتوانید لاگ دقیق‌تری اضافه کنید
            for db_name, client in self.databases.items():
                if client is None:
                    print(f"   ❌ {db_name.upper()} failed to connect")
    
    def get_client(self, db_name: str) -> Optional[redis.Redis]:
        """دریافت client دیتابیس مورد نظر"""
        return self.databases.get(db_name)
    
    def set(self, db_name: str, key: str, value: Any, expire: int = 300) -> Tuple[bool, float]:
        """ذخیره داده در دیتابیس مشخص - بازگشت (موفقیت, زمان پاسخ)"""
        client = self.get_client(db_name)
        if not client:
            return False, 0
        
        try:
            start_time = time.time()
            serialized_value = json.dumps(value, ensure_ascii=False)
            success = bool(client.setex(key, expire, serialized_value))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            print(f"Redis set error for db {db_name}, key {key}: {e}")
            return False, 0
    
    def get(self, db_name: str, key: str) -> Tuple[Optional[Any], float]:
        """دریافت داده از دیتابیس مشخص - بازگشت (داده, زمان پاسخ)"""
        client = self.get_client(db_name)
        if not client:
            return None, 0
        
        try:
            start_time = time.time()
            value = client.get(key)
            response_time = time.time() - start_time
            
            if value:
                data = json.loads(value)
                return data, response_time
            else:
                return None, response_time
        except Exception as e:
            print(f"Redis get error for db {db_name}, key {key}: {e}")
            return None, 0
    
    def delete(self, db_name: str, key: str) -> Tuple[bool, float]:
        """حذف داده از دیتابیس مشخص - بازگشت (موفقیت, زمان پاسخ)"""
        client = self.get_client(db_name)
        if not client:
            return False, 0
        
        try:
            start_time = time.time()
            success = bool(client.delete(key))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            print(f"Redis delete error for db {db_name}, key {key}: {e}")
            return False, 0
    
    def exists(self, db_name: str, key: str) -> Tuple[bool, float]:
        """بررسی وجود کلید در دیتابیس مشخص"""
        client = self.get_client(db_name)
        if not client:
            return False, 0
        
        try:
            start_time = time.time()
            exists = bool(client.exists(key))
            response_time = time.time() - start_time
            return exists, response_time
        except Exception as e:
            print(f"Redis exists error for db {db_name}, key {key}: {e}")
            return False, 0
    
    def get_keys(self, db_name: str, pattern: str = "*") -> Tuple[List[str], float]:
        """دریافت کلیدها از دیتابیس مشخص"""
        client = self.get_client(db_name)
        if not client:
            return [], 0
        
        try:
            start_time = time.time()
            keys = client.keys(pattern)
            response_time = time.time() - start_time
            return keys, response_time
        except Exception as e:
            print(f"Redis keys error for db {db_name}, pattern {pattern}: {e}")
            return [], 0
    
    def set_compressed(self, db_name: str, key: str, value: Any, expire: int = 300) -> Tuple[bool, float]:
        """ذخیره داده فشرده شده (برای داده‌های حجیم در UTC)"""
        import gzip
        client = self.get_client(db_name)
        if not client:
            return False, 0
        
        try:
            start_time = time.time()
            serialized_value = json.dumps(value, ensure_ascii=False)
            compressed_value = gzip.compress(serialized_value.encode('utf-8'))
            success = bool(client.setex(key, expire, compressed_value))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            print(f"Redis set_compressed error for db {db_name}, key {key}: {e}")
            return False, 0
    
    def get_compressed(self, db_name: str, key: str) -> Tuple[Optional[Any], float]:
        """دریافت داده فشرده شده"""
        import gzip
        client = self.get_client(db_name)
        if not client:
            return None, 0
        
        try:
            start_time = time.time()
            value = client.get(key)
            response_time = time.time() - start_time
            
            if value:
                decompressed_value = gzip.decompress(value).decode('utf-8')
                data = json.loads(decompressed_value)
                return data, response_time
            else:
                return None, response_time
        except Exception as e:
            print(f"Redis get_compressed error for db {db_name}, key {key}: {e}")
            return None, 0
    
    def health_check(self, db_name: str = None) -> Dict[str, Any]:
        """بررسی سلامت دیتابیس‌ها"""
        if db_name:
            return self._single_health_check(db_name)
        else:
            health_report = {}
            for db in self.databases.keys():
                health_report[db] = self._single_health_check(db)
            return health_report
    
    def _single_health_check(self, db_name: str) -> Dict[str, Any]:
        """بررسی سلامت یک دیتابیس - نسخه اصلاح شده"""
        client = self.get_client(db_name)
        if not client:
            return {
                "status": "disconnected", 
                "database": db_name,
                "storage_type": "cloud",
                "error": "No Redis client available",
                "timestamp": datetime.now().isoformat()
            }
    
        try:
            start_time = time.time()
            client.ping()
            ping_time = time.time() - start_time
        
            info = client.info()
            used_memory = info.get('used_memory', 0)
            max_memory = 256 * 1024 * 1024  # 256MB برای هر دیتابیس ابری
        
            return {
                "status": "connected",
                "database": db_name,
                "storage_type": "cloud",
                "ping_time_ms": round(ping_time * 1000, 2),
                "max_memory_mb": 256,
                "used_memory_mb": round(used_memory / (1024 * 1024), 2),
                "used_memory_percent": round((used_memory / max_memory) * 100, 2),
                "available_mb": round(256 - (used_memory / (1024 * 1024)), 2),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_ratio": round(info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)), 4),
                "uptime_in_seconds": info.get('uptime_in_seconds', 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error", 
                "database": db_name,
                "storage_type": "cloud",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_database_usage(self) -> Dict[str, Dict[str, Any]]:
        """دریافت گزارش استفاده از هر دیتابیس - نسخه Hybrid"""
        usage_report = {}
        for db_name, client in self.databases.items():
            if client:
                try:
                    info = client.info()
                    used_memory = info.get('used_memory', 0)
                    max_memory = 256 * 1024 * 1024  # 256MB برای دیتابیس ابری
                
                    usage_report[db_name] = {
                        'storage_type': 'cloud',
                        'max_memory_mb': 256,
                        'used_memory_mb': round(used_memory / (1024 * 1024), 2),
                        'used_memory_percentage': round((used_memory / max_memory) * 100, 2),
                        'available_mb': round(256 - (used_memory / (1024 * 1024)), 2),
                        'keys_count': sum([int(info.get(f'db{i}', {}).get('keys', 0)) for i in range(16)]),
                        'connected_clients': info.get('connected_clients', 0),
                        'hit_ratio': round(info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)) * 100, 2)
                    }
                except Exception as e:
                    usage_report[db_name] = {
                        'storage_type': 'cloud',
                        'error': str(e),
                        'max_memory_mb': 256
                    }
            else:
                usage_report[db_name] = {
                    'storage_type': 'cloud', 
                    'error': 'Client not connected',
                    'max_memory_mb': 256
                }
        return usage_report
# نمونه global برای استفاده در سایر فایل‌ها
redis_manager = RedisCacheManager()
