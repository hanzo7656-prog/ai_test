import redis
import json
import os
import time
from datetime import datetime
from typing import Any, Optional, Tuple, List, Dict

class RedisCacheManager:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.client = None
        self._connect()
        
    def _connect(self):
        """اتصال به Redis"""
        try:
            self.client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=10
            )
            self.client.ping()
            print("✅ Redis Manager connected successfully!")
        except Exception as e:
            print(f"❌ Redis Manager connection failed: {e}")
            self.client = None
    
    def set(self, key: str, value: Any, expire: int = 300) -> Tuple[bool, float]:
        """ذخیره داده در کش - بازگشت (موفقیت, زمان پاسخ)"""
        if not self.client:
            return False, 0
        
        try:
            start_time = time.time()
            serialized_value = json.dumps(value, ensure_ascii=False)
            success = bool(self.client.setex(key, expire, serialized_value))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            print(f"Redis set error for key {key}: {e}")
            return False, 0
    
    def get(self, key: str) -> Tuple[Optional[Any], float]:
        """دریافت داده از کش - بازگشت (داده, زمان پاسخ)"""
        if not self.client:
            return None, 0
        
        try:
            start_time = time.time()
            value = self.client.get(key)
            response_time = time.time() - start_time
            
            if value:
                data = json.loads(value)
                return data, response_time
            else:
                return None, response_time
        except Exception as e:
            print(f"Redis get error for key {key}: {e}")
            return None, 0
    
    def delete(self, key: str) -> Tuple[bool, float]:
        """حذف داده از کش - بازگشت (موفقیت, زمان پاسخ)"""
        if not self.client:
            return False, 0
        
        try:
            start_time = time.time()
            success = bool(self.client.delete(key))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            print(f"Redis delete error for key {key}: {e}")
            return False, 0
    
    def exists(self, key: str) -> Tuple[bool, float]:
        """بررسی وجود کلید - بازگشت (وجود دارد, زمان پاسخ)"""
        if not self.client:
            return False, 0
        
        try:
            start_time = time.time()
            exists = bool(self.client.exists(key))
            response_time = time.time() - start_time
            return exists, response_time
        except Exception as e:
            print(f"Redis exists error for key {key}: {e}")
            return False, 0
    
    def get_keys(self, pattern: str = "*") -> Tuple[List[str], float]:
        """دریافت کلیدها - بازگشت (لیست کلیدها, زمان پاسخ)"""
        if not self.client:
            return [], 0
        
        try:
            start_time = time.time()
            keys = self.client.keys(pattern)
            response_time = time.time() - start_time
            return keys, response_time
        except Exception as e:
            print(f"Redis keys error for pattern {pattern}: {e}")
            return [], 0
    
    def health_check(self) -> Dict[str, Any]:
        """بررسی سلامت کامل Redis"""
        if not self.client:
            return {
                "status": "disconnected", 
                "error": "No Redis client available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            start_time = time.time()
            self.client.ping()
            ping_time = time.time() - start_time
            
            info = self.client.info()
            return {
                "status": "connected",
                "type": "redis_cloud",
                "ping_time_ms": round(ping_time * 1000, 2),
                "used_memory": info.get('used_memory_human', 'N/A'),
                "used_memory_bytes": info.get('used_memory', 0),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "uptime_in_seconds": info.get('uptime_in_seconds', 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
