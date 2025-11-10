import redis
import json
import os
from typing import Any, Optional

class RedisManager:
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
                socket_timeout=5
            )
            # تست اتصال
            self.client.ping()
            print("✅ Redis Cloud connected successfully!")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.client = None
    
    def set(self, key: str, value: Any, expire: int = 300) -> bool:
        """ذخیره داده در کش"""
        if not self.client:
            return False
        try:
            serialized_value = json.dumps(value)
            return self.client.setex(key, expire, serialized_value)
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """دریافت داده از کش"""
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """حذف داده از کش"""
        if not self.client:
            return False
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def health_check(self) -> dict:
        """بررسی سلامت Redis"""
        if not self.client:
            return {"status": "disconnected", "error": "No Redis client"}
        try:
            self.client.ping()
            return {"status": "connected", "type": "redis_cloud"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# نمونه گلوبال
redis_manager = RedisManager()
