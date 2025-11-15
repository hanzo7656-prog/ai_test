# config.py - فایل تنظیمات کامل VortexAI API
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """تنظیمات اصلی برنامه"""
    
    # ==================== BASIC CONFIGURATION ====================
    APP_NAME: str = "VortexAI API"
    APP_VERSION: str = "4.0.0"
    APP_DESCRIPTION: str = "Complete Crypto AI System with Advanced Debugging"
    
    # ==================== SERVER CONFIGURATION ====================
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 10000))
    DEBUG: bool = False
    RELOAD: bool = False
    
    # ==================== API CONFIGURATION ====================
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "VortexAI Crypto API"
    
    # ==================== EXTERNAL APIS CONFIGURATION ====================
    COINSTATS_API_KEY: str = "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="
    COINSTATS_BASE_URL: str = "https://openapiv1.coinstats.app"
    
    # ==================== CACHE CONFIGURATION ====================
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL: int = 300  # 5 minutes
    CACHE_DIR: str = "./coinstats_cache"
    
    # Redis Configuration for Hybrid Architecture
    REDIS_CONFIG: Dict[str, Dict] = {
        'uta': {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'role': 'AI Core Models - Long term storage',
            'max_memory_mb': 256
        },
        'utb': {
            'host': 'localhost', 
            'port': 6379,
            'db': 1,
            'role': 'AI Processed Data - Medium TTL',
            'max_memory_mb': 256
        },
        'utc': {
            'host': 'localhost',
            'port': 6379,
            'db': 2,
            'role': 'Raw Data + Historical Archive',
            'max_memory_mb': 256
        },
        'mother_a': {
            'host': 'localhost',
            'port': 6379,
            'db': 3,
            'role': 'System Core Data',
            'max_memory_mb': 256
        },
        'mother_b': {
            'host': 'localhost',
            'port': 6379,
            'db': 4,
            'role': 'Operations & Analytics',
            'max_memory_mb': 256
        }
    }
    
    # ==================== DATABASE CONFIGURATION ====================
    DATABASE_URL: str = "sqlite:///./vortexai.db"
    
    # ==================== SECURITY CONFIGURATION ====================
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # ==================== RATE LIMITING CONFIGURATION ====================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_COOLDOWN_MINUTES: int = 5
    
    # CoinStats API Rate Limiting
    COINSTATS_RATE_LIMIT_INTERVAL: float = 0.2  # 200ms between requests
    
    # ==================== LOGGING CONFIGURATION ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "vortexai.log"
    
    # ==================== PERFORMANCE CONFIGURATION ====================
    MAX_WORKERS: int = 10
    BACKGROUND_TASK_TIMEOUT: int = 300  # 5 minutes
    REQUEST_TIMEOUT: int = 30  # seconds
    
    # ==================== DEBUG SYSTEM CONFIGURATION ====================
    DEBUG_SYSTEM_ENABLED: bool = True
    DEBUG_METRICS_RETENTION_DAYS: int = 7
    DEBUG_MAX_ENDPOINT_CALLS: int = 10000
    DEBUG_MAX_SYSTEM_METRICS: int = 1000
    
    # Performance Thresholds
    PERFORMANCE_THRESHOLDS: Dict[str, float] = {
        'response_time_warning': 1.0,      # seconds
        'response_time_critical': 3.0,     # seconds
        'cpu_warning': 80.0,               # percent
        'cpu_critical': 95.0,              # percent  
        'memory_warning': 85.0,            # percent
        'memory_critical': 95.0            # percent
    }
    
    # ==================== CACHE STRATEGIES CONFIGURATION ====================
    CACHE_STRATEGIES: Dict[str, Dict] = {
        "processed_data": {
            "coins": {
                "realtime_ttl": 600,           # 10 minutes
                "archive_ttl": 31536000,       # 1 year
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
                "archive_ttl": 31536000,       # 1 year
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
                "archive_ttl": 2592000,        # 30 days
                "strategy": "hourly", 
                "database": "utc"
            },
            "raw_news": {
                "realtime_ttl": 300,           # 5 minutes
                "archive_ttl": 7776000,        # 90 days
                "strategy": "daily",
                "database": "utc"
            },
            "raw_insights": {
                "realtime_ttl": 900,           # 15 minutes
                "archive_ttl": 15552000,       # 6 months
                "strategy": "daily",
                "database": "utc"
            },
            "raw_exchanges": {
                "realtime_ttl": 300,           # 5 minutes
                "archive_ttl": 2592000,        # 30 days
                "strategy": "hourly",
                "database": "utc"
            }
        }
    }
    
    # ==================== HEALTH CHECK CONFIGURATION ====================
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    HEALTH_CHECK_TIMEOUT: int = 10   # seconds
    
    # ==================== ARCHIVE SYSTEM CONFIGURATION ====================
    ARCHIVE_ENABLED: bool = True
    ARCHIVE_CLEANUP_DAYS: int = 365  # Clean archives older than 1 year
    ARCHIVE_STRATEGIES: List[str] = ["hourly", "daily", "weekly"]
    
    # ==================== RESOURCE MANAGEMENT CONFIGURATION ====================
    # Disk Space Management (for 1GB environment)
    DISK_CLEANUP_THRESHOLD: int = 85  # percent
    DISK_CLEANUP_URGENT_THRESHOLD: int = 90  # percent
    
    # Memory Management
    MEMORY_CLEANUP_THRESHOLD: int = 80  # percent
    
    # ==================== BACKGROUND WORKER CONFIGURATION ====================
    BACKGROUND_WORKER_ENABLED: bool = True
    BACKGROUND_WORKER_MAX_TASKS: int = 100
    BACKGROUND_WORKER_POLL_INTERVAL: int = 5  # seconds
    
    # ==================== WEBSOCKET CONFIGURATION ====================
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_PING_INTERVAL: int = 20  # seconds
    WEBSOCKET_PING_TIMEOUT: int = 10   # seconds
    
    # ==================== EXTERNAL SERVICES CONFIGURATION ====================
    # Available external services
    EXTERNAL_SERVICES: Dict[str, bool] = {
        "coinstats_api": True,
        "redis_cache": True,
        "debug_system": True,
        "ai_system": False,  # Currently not available based on files
        "background_worker": True
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
    
    # ==================== ENVIRONMENT SPECIFIC ====================
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# ایجاد نمونه تنظیمات
settings = Settings()

# ==================== VALIDATION FUNCTIONS ====================

def validate_config() -> Dict[str, Any]:
    """اعتبارسنجی تنظیمات و برگرداندن وضعیت"""
    validation_results = {
        "basic_config": True,
        "api_config": True,
        "cache_config": True,
        "security_config": True,
        "performance_config": True
    }
    
    try:
        # اعتبارسنجی تنظیمات پایه
        if not settings.APP_NAME or not settings.APP_VERSION:
            validation_results["basic_config"] = False
            logger.error("❌ Basic configuration validation failed")
        
        # اعتبارسنجی API configuration
        if not settings.COINSTATS_API_KEY or settings.COINSTATS_API_KEY == "your-api-key-here":
            validation_results["api_config"] = False
            logger.error("❌ API configuration validation failed")
        
        # اعتبارسنجی تنظیمات کش
        if settings.CACHE_ENABLED and not settings.REDIS_CONFIG:
            validation_results["cache_config"] = False
            logger.error("❌ Cache configuration validation failed")
        
        # اعتبارسنجی امنیت
        if settings.SECRET_KEY == "your-secret-key-here-change-in-production":
            logger.warning("⚠️  Using default secret key - change in production")
        
        # اعتبارسنجی عملکرد
        if settings.MAX_WORKERS <= 0 or settings.REQUEST_TIMEOUT <= 0:
            validation_results["performance_config"] = False
            logger.error("❌ Performance configuration validation failed")
            
    except Exception as e:
        logger.error(f"❌ Configuration validation error: {e}")
        for key in validation_results:
            validation_results[key] = False
    
    return validation_results

def get_config_summary() -> Dict[str, Any]:
    """دریافت خلاصه‌ای از تنظیمات"""
    validation = validate_config()
    
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        },
        "server": {
            "host": settings.HOST,
            "port": settings.PORT,
            "debug": settings.DEBUG
        },
        "features": {
            "cache_enabled": settings.CACHE_ENABLED,
            "debug_system_enabled": settings.DEBUG_SYSTEM_ENABLED,
            "background_worker_enabled": settings.BACKGROUND_WORKER_ENABLED,
            "websocket_enabled": settings.WEBSOCKET_ENABLED
        },
        "external_services": settings.EXTERNAL_SERVICES,
        "validation": validation,
        "config_valid": all(validation.values()),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

# ==================== ENVIRONMENT SPECIFIC OVERRIDES ====================

def apply_environment_overrides():
    """اعمال تنظیمات خاص محیط"""
    if settings.ENVIRONMENT == "production":
        settings.DEBUG = False
        settings.RELOAD = False
        settings.LOG_LEVEL = "INFO"
        
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
    "description": settings.APP_DESCRIPTION
}

SERVER_CONFIG = {
    "host": settings.HOST,
    "port": settings.PORT,
    "debug": settings.DEBUG,
    "reload": settings.RELOAD
}

API_CONFIG = {
    "coinstats_api_key": settings.COINSTATS_API_KEY,
    "coinstats_base_url": settings.COINSTATS_BASE_URL,
    "rate_limit_interval": settings.COINSTATS_RATE_LIMIT_INTERVAL
}

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
    print("🚀 VortexAI Configuration Summary:")
    print(f"   App: {summary['app']['name']} v{summary['app']['version']}")
    print(f"   Environment: {summary['app']['environment']}")
    print(f"   Server: {summary['server']['host']}:{summary['server']['port']}")
    print(f"   Features: Cache({summary['features']['cache_enabled']}), "
          f"Debug({summary['features']['debug_system_enabled']}), "
          f"WebSocket({summary['features']['websocket_enabled']})")
    print(f"   Config Valid: {summary['config_valid']}")
