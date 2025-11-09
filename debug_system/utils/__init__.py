"""
VortexAI Utility Modules
Common utilities and helpers for the VortexAI system
"""

from .config_loader import ConfigLoader, config_loader
from .logger import setup_logging, get_logger, JSONFormatter, ColoredFormatter
from .error_handler import ErrorHandler, APIError, error_handler
from .data_normalizer import DataNormalizer, data_normalizer  # ✅ اضافه شد

# راه‌اندازی utilityها
def initialize_utilities():
    """راه‌اندازی سیستم utilityها"""
    try:
        # سیستم لاگینگ قبلاً در logger.py راه‌اندازی شده
        # سیستم مدیریت خطا قبلاً راه‌اندازی شده
        # سیستم کانفیگ قبلاً راه‌اندازی شده
        # سیستم نرمال‌سازی داده‌ها اضافه شد
        
        print("✅ Utility systems initialized:")
        print(f"   - Config Loader: {type(config_loader).__name__}")
        print(f"   - Error Handler: {type(error_handler).__name__}")
        print(f"   - Logging System: Active (JSON + Colored Console)")
        print(f"   - Data Normalizer: {type(data_normalizer).__name__}")  # ✅ اضافه شد
        
        return {
            "config_loader": config_loader,
            "error_handler": error_handler,
            "data_normalizer": data_normalizer  # ✅ اضافه شد
        }
    except Exception as e:
        print(f"❌ Utilities initialization error: {e}")
        raise

# راه‌اندازی خودکار
utilities_system = initialize_utilities()

__all__ = [
    "ConfigLoader", "config_loader",
    "setup_logging", "get_logger", "JSONFormatter", "ColoredFormatter", 
    "ErrorHandler", "APIError", "error_handler",
    "DataNormalizer", "data_normalizer",  # ✅ اضافه شد
    "initialize_utilities", "utilities_system"
]
