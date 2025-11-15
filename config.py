# config.py - فایل تنظیمات واقعی VortexAI API
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """تنظیمات اصلی برنامه - مبتنی بر اطلاعات واقعی"""
    
    # ==================== BASIC CONFIGURATION ====================
    APP_NAME: str = "VortexAI API"
    APP_VERSION: str = "4.0.0"
    APP_DESCRIPTION: str = "Complete Crypto AI System with Advanced Debugging"
    
    # ==================== SERVER CONFIGURATION ====================
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 10000))
    DEBUG: bool = False
    RELOAD: bool = False
    SERVICE_URL: str = os.environ.get("SERVICE_URL", "")
    
    # ==================== ENVIRONMENT CONFIGURATION ====================
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "production")
    PYTHON_VERSION: str = os.environ.get("PYTHON_VERSION", "3.9.0")
    
    # ==================== API CONFIGURATION ====================
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "VortexAI Crypto API"
    
    # ==================== EXTERNAL APIS CONFIGURATION ====================
    COINSTATS_API_KEY: str = os.environ.get("COINSTATS_API_KEY", "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=")
    COINSTATS_BASE_URL: str = "https://openapiv1.coinstats.app"
    
    # ==================== REDIS CONFIGURATION (Upstash) ====================
    # Redis URLs from Render environment variables
    REDIS_URLS: Dict[str, str] = {
        'uta': os.environ.get("UTA_REDIS_AI", ""),
        'utb': os.environ.get("UTB_REDIS_AI", ""),
        'utc': os.environ.get("UTC_REDIS_AI", ""),
        'mother_a': os.environ.get("MOTHER_A_URL", ""),
        'mother_b': os.environ.get("MOTHER_B_URL", "")
    }
    
    # Redis Configuration for Hybrid Architecture
    REDIS_CONFIG: Dict[str, Dict] = {
        'uta': {
            'url': os.environ.get("UTA_REDIS_AI", ""),
            'role': 'AI Core Models - Long term storage',
            'max_memory_mb': 256,
            'database': 0
        },
        'utb': {
            'url': os.environ.get("UTB_REDIS_AI", ""),
            'role': 'AI Processed Data - Medium TTL',
            'max_memory_mb': 256,
            'database': 1
        },
        'utc': {
            'url': os.environ.get("UTC_REDIS_AI", ""),
            'role': 'Raw Data + Historical Archive',
            'max_memory_mb': 256,
            'database': 2
        },
        'mother_a': {
            'url': os.environ.get("MOTHER_A_URL", ""),
            'role': 'System Core Data',
            'max_memory_mb': 256,
            'database': 3
        },
        'mother_b': {
            'url': os.environ.get("MOTHER_B_URL", ""),
            'role': 'Operations & Analytics',
            'max_memory_mb': 256,
            'database': 4
        }
    }
    
    # ==================== CACHE CONFIGURATION ====================
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL: int = 300  # 5 minutes
    CACHE_DIR: str = "./coinstats_cache"
    
    # ==================== RESOURCE LIMITS (Render Specifications) ====================
    RESOURCE_LIMITS: Dict[str, Any] = {
        'memory_mb': 512,           # حافظه واقعی سرور Render
        'disk_gb': 1,               # دیسک 1 گیگابایت
        'bandwidth_gb': 100,        # پهنای باند 100 گیگابایت
        'cpu_cores': 0.1,           # CPU Render
        'total_redis_storage_mb': 1280  # مجموع فضای Redis
    }
    
    # ==================== SECURITY CONFIGURATION ====================
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "vortexai-default-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "https://vortexai-api.onrender.com",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"  # برای توسعه
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # ==================== RATE LIMITING CONFIGURATION ====================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100  # با توجه به محدودیت‌های Render
    RATE_LIMIT_COOLDOWN_MINUTES: int = 5
    
    # CoinStats API Rate Limiting
    COINSTATS_RATE_LIMIT_INTERVAL: float = 0.2  # 200ms between requests
    
    # ==================== LOGGING CONFIGURATION ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "vortexai.log"
    
    # ==================== PERFORMANCE CONFIGURATION ====================
    MAX_WORKERS: int = 4  # با توجه به 0.1 CPU
    BACKGROUND_TASK_TIMEOUT: int = 180  # 3 minutes - کاهش به دلیل محدودیت CPU
    REQUEST_TIMEOUT: int = 30  # seconds
    
    # ==================== DEBUG SYSTEM CONFIGURATION ====================
    DEBUG_SYSTEM_ENABLED: bool = True
    DEBUG_METRICS_RETENTION_DAYS: int = 7
    DEBUG_MAX_ENDPOINT_CALLS: int = 5000  # کاهش به دلیل محدودیت حافظه
    DEBUG_MAX_SYSTEM_METRICS: int = 500   # کاهش به دلیل محدودیت حافظه
    
    # Performance Thresholds - تنظیم شده برای منابع Render
    PERFORMANCE_THRESHOLDS: Dict[str, float] = {
        'response_time_warning': 2.0,      # افزایش به دلیل CPU کم
        'response_time_critical': 5.0,     # افزایش به دلیل CPU کم
        'cpu_warning': 70.0,               # کاهش به دلیل CPU کم
        'cpu_critical': 85.0,              # کاهش به دلیل CPU کم
        'memory_warning': 75.0,            # کاهش به دلیل حافظه کم
        'memory_critical': 85.0,           # کاهش به دلیل حافظه کم
        'disk_warning': 80.0,              # هشدار دیسک
        'disk_critical': 90.0              # بحران دیسک
    }
    
    # ==================== CACHE STRATEGIES CONFIGURATION ====================
    CACHE_STRATEGIES: Dict[str, Dict] = {
        "processed_data": {
            "coins": {
                "realtime_ttl": 600,           # 10 minutes
                "archive_ttl": 2592000,        # 30 days - کاهش به دلیل فضای محدود
                "strategy": "daily",
                "database": "utb"
            },
            "news": {
                "realtime_ttl": 600,           # 10 minutes  
                "archive_ttl": 15552000,       # 6 months
                "strategy": "weekly", 
                "database": "utb"
            },
            "insights": {
                "realtime_ttl": 3600,          # 1 hour
                "archive_ttl": 2592000,        # 30 days - کاهش
                "strategy": "weekly",
                "database": "utb"
            },
            "exchanges": {
                "realtime_ttl": 600,           # 10 minutes
                "archive_ttl": 15552000,       # 6 months
                "strategy": "daily",
                "database": "utb"
            }
        },
        "raw_data": {
            "raw_coins": {
                "realtime_ttl": 180,           # 3 minutes
                "archive_ttl": 604800,         # 7 days - کاهش شدید
                "strategy": "hourly", 
                "database": "utc"
            },
            "raw_news": {
                "realtime_ttl": 300,           # 5 minutes
                "archive_ttl": 2592000,        # 30 days
                "strategy": "daily",
                "database": "utc"
            },
            "raw_insights": {
                "realtime_ttl": 900,           # 15 minutes
                "archive_ttl": 604800,         # 7 days - کاهش
                "strategy": "daily",
                "database": "utc"
            },
            "raw_exchanges": {
                "realtime_ttl": 300,           # 5 minutes
                "archive_ttl": 604800,         # 7 days - کاهش
                "strategy": "hourly",
                "database": "utc"
            }
        }
    }
    
    # ==================== HEALTH CHECK CONFIGURATION ====================
    HEALTH_CHECK_INTERVAL: int = 60  # افزایش به دلیل CPU کم
    HEALTH_CHECK_TIMEOUT: int = 10   # seconds
    
    # ==================== ARCHIVE SYSTEM CONFIGURATION ====================
    ARCHIVE_ENABLED: bool = True
    ARCHIVE_CLEANUP_DAYS: int = 30  # کاهش به 30 روز به دلیل فضای محدود
    ARCHIVE_STRATEGIES: List[str] = ["daily", "weekly"]  # حذف hourly
    
    # ==================== RESOURCE MANAGEMENT CONFIGURATION ====================
    # Disk Space Management (برای محیط 1GB)
    DISK_CLEANUP_THRESHOLD: int = 80   # درصد
    DISK_CLEANUP_URGENT_THRESHOLD: int = 85  # درصد
    
    # Memory Management - تنظیم شده برای 512MB
    MEMORY_CLEANUP_THRESHOLD: int = 70  # درصد
    
    # ==================== BACKGROUND WORKER CONFIGURATION ====================
    BACKGROUND_WORKER_ENABLED: bool = True
    BACKGROUND_WORKER_MAX_TASKS: int = 10   # کاهش به دلیل منابع محدود
    BACKGROUND_WORKER_POLL_INTERVAL: int = 10  # افزایش به دلیل CPU کم
    
    # ==================== WEBSOCKET CONFIGURATION ====================
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_PING_INTERVAL: int = 30  # افزایش به دلیل CPU کم
    WEBSOCKET_PING_TIMEOUT: int = 15   # seconds
    
    # ==================== EXTERNAL SERVICES CONFIGURATION ====================
    # Available external services
    EXTERNAL_SERVICES: Dict[str, bool] = {
        "coinstats_api": True,
        "redis_cache": True,
        "debug_system": True,
        "ai_system": False,
        "background_worker": True,
        "upstash_redis": True
    }
    
    # ==================== ROUTES CONFIGURATION ====================
    ENABLED_ROUTES: Dict[str, bool] = {
        "health": True,
        "coins": True,
        "exchanges": True,
        "news": True,
        "insights": True,
        "raw_coins": True,
        "raw_news": True,
        "raw_insights": True,
        "raw_exchanges": True,
        "docs": True,
        "debug": True
    }

    class Config:
        case_sensitive = True
        env_file = ".env"

# ایجاد نمونه تنظیمات
settings = Settings()

# ==================== VALIDATION FUNCTIONS ====================

def validate_redis_connections() -> Dict[str, bool]:
    """اعتبارسنجی اتصالات Redis"""
    connection_status = {}
    
    for db_name, config in settings.REDIS_CONFIG.items():
        url = config.get('url', '')
        if url and url.startswith('redis://'):
            connection_status[db_name] = True
            logger.info(f"✅ Redis {db_name} configured")
        else:
            connection_status[db_name] = False
            logger.warning(f"⚠️ Redis {db_name} not properly configured")
    
    return connection_status

def validate_config() -> Dict[str, Any]:
    """اعتبارسنجی کامل تنظیمات"""
    validation_results = {
        "basic_config": True,
        "api_config": True,
        "redis_config": True,
        "resource_config": True,
        "security_config": True
    }
    
    try:
        # اعتبارسنجی تنظیمات پایه
        if not settings.APP_NAME or not settings.APP_VERSION:
            validation_results["basic_config"] = False
            logger.error("❌ Basic configuration validation failed")
        
        # اعتبارسنجی API configuration
        if not settings.COINSTATS_API_KEY:
            validation_results["api_config"] = False
            logger.error("❌ CoinStats API key not configured")
        
        # اعتبارسنجی Redis
        redis_status = validate_redis_connections()
        connected_dbs = sum(1 for status in redis_status.values() if status)
        if connected_dbs == 0:
            validation_results["redis_config"] = False
            logger.error("❌ No Redis databases configured")
        elif connected_dbs < 5:
            logger.warning(f"⚠️ Only {connected_dbs}/5 Redis databases configured")
        
        # اعتبارسنجی منابع
        if settings.RESOURCE_LIMITS['memory_mb'] <= 0:
            validation_results["resource_config"] = False
            logger.error("❌ Invalid memory configuration")
            
    except Exception as e:
        logger.error(f"❌ Configuration validation error: {e}")
        for key in validation_results:
            validation_results[key] = False
    
    return validation_results

def get_config_summary() -> Dict[str, Any]:
    """دریافت خلاصه‌ای از تنظیمات واقعی"""
    validation = validate_config()
    redis_status = validate_redis_connections()
    
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "python_version": settings.PYTHON_VERSION,
            "service_url": settings.SERVICE_URL
        },
        "server": {
            "host": settings.HOST,
            "port": settings.PORT,
            "debug": settings.DEBUG
        },
        "resources": {
            "memory_mb": settings.RESOURCE_LIMITS['memory_mb'],
            "disk_gb": settings.RESOURCE_LIMITS['disk_gb'],
            "cpu_cores": settings.RESOURCE_LIMITS['cpu_cores'],
            "bandwidth_gb": settings.RESOURCE_LIMITS['bandwidth_gb'],
            "redis_storage_mb": settings.RESOURCE_LIMITS['total_redis_storage_mb']
        },
        "redis": {
            "connected_databases": sum(1 for status in redis_status.values() if status),
            "total_databases": len(redis_status),
            "status": redis_status
        },
        "features": {
            "cache_enabled": settings.CACHE_ENABLED,
            "debug_system_enabled": settings.DEBUG_SYSTEM_ENABLED,
            "background_worker_enabled": settings.BACKGROUND_WORKER_ENABLED,
            "websocket_enabled": settings.WEBSOCKET_ENABLED
        },
        "validation": validation,
        "config_valid": all(validation.values()),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

# ==================== ENVIRONMENT SPECIFIC OVERRIDES ====================

def apply_environment_overrides():
    """اعمال تنظیمات خاص محیط Render"""
    if settings.ENVIRONMENT == "production":
        settings.DEBUG = False
        settings.RELOAD = False
        settings.LOG_LEVEL = "INFO"
        # محدودیت‌های بیشتر در production
        settings.MAX_WORKERS = 2
        settings.BACKGROUND_WORKER_MAX_TASKS = 5
        
    elif settings.ENVIRONMENT == "development":
        settings.DEBUG = True
        settings.RELOAD = True
        settings.LOG_LEVEL = "DEBUG"
        
    elif settings.ENVIRONMENT == "testing":
        settings.DEBUG = True
        settings.RELOAD = False
        settings.LOG_LEVEL = "WARNING"
        settings.CACHE_ENABLED = False

# اعمال تنظیمات محیط در زمان ایمپورت
apply_environment_overrides()

# ==================== CONFIGURATION EXPORTS ====================

# Export برای استفاده آسان در سایر ماژول‌ها
APP_CONFIG = {
    "name": settings.APP_NAME,
    "version": settings.APP_VERSION,
    "description": settings.APP_DESCRIPTION,
    "environment": settings.ENVIRONMENT,
    "python_version": settings.PYTHON_VERSION
}

SERVER_CONFIG = {
    "host": settings.HOST,
    "port": settings.PORT,
    "debug": settings.DEBUG,
    "reload": settings.RELOAD,
    "service_url": settings.SERVICE_URL
}

API_CONFIG = {
    "coinstats_api_key": settings.COINSTATS_API_KEY,
    "coinstats_base_url": settings.COINSTATS_BASE_URL,
    "rate_limit_interval": settings.COINSTATS_RATE_LIMIT_INTERVAL
}

REDIS_CONFIG = settings.REDIS_CONFIG
RESOURCE_CONFIG = settings.RESOURCE_LIMITS

CACHE_CONFIG = {
    "enabled": settings.CACHE_ENABLED,
    "default_ttl": settings.CACHE_DEFAULT_TTL,
    "cache_dir": settings.CACHE_DIR,
    "redis_config": settings.REDIS_CONFIG,
    "strategies": settings.CACHE_STRATEGIES
}

DEBUG_CONFIG = {
    "enabled": settings.DEBUG_SYSTEM_ENABLED,
    "metrics_retention_days": settings.DEBUG_METRICS_RETENTION_DAYS,
    "max_endpoint_calls": settings.DEBUG_MAX_ENDPOINT_CALLS,
    "max_system_metrics": settings.DEBUG_MAX_SYSTEM_METRICS,
    "performance_thresholds": settings.PERFORMANCE_THRESHOLDS
}

# لاگ خلاصه تنظیمات در زمان راه‌اندازی
if __name__ == "__main__":
    summary = get_config_summary()
    print("🚀 VortexAI Real Configuration Summary:")
    print(f"   App: {summary['app']['name']} v{summary['app']['version']}")
    print(f"   Environment: {summary['app']['environment']} (Python {summary['app']['python_version']})")
    print(f"   Server: {summary['server']['host']}:{summary['server']['port']}")
    print(f"   Resources: {summary['resources']['memory_mb']}MB RAM, {summary['resources']['cpu_cores']} CPU")
    print(f"   Redis: {summary['redis']['connected_databases']}/{summary['redis']['total_databases']} databases connected")
    print(f"   Config Valid: {summary['config_valid']}")
