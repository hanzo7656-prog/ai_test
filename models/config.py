# config.py
import os
from datetime import timedelta

# تنظیمات بهینه‌سازی حافظه
MEMORY_OPTIMIZATION = {
    'dtype_optimization': True,
    'downcast_numbers': True,
    'remove_unused_columns': True,
    'chunk_processing': True,
    'chunk_size': 1000,
    'max_data_points': 5000,  # حداکثر نقطه داده برای پردازش
    'cleanup_interval': 100   # هر 100 پردازش، حافظه پاک‌سازی شود
}

# تنظیمات اندیکاتورها
INDICATOR_CONFIG = {
    'enable_basic_indicators': True,
    'enable_advanced_indicators': False,  # غیرفعال برای صرفه‌جویی حافظه
    'rsi_period': 14,
    'sma_periods': [20, 50],  # فقط میانگین‌های ضروری
    'bb_period': 20,
    'max_indicators': 10  # حداکثر تعداد اندیکاتورها
}

# تنظیمات API
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'timeout': 30,
    'retry_attempts': 3,
    'cache_ttl': timedelta(minutes=10)
}

# تنظیمات هوش مصنوعی
AI_CONFIG = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'sma_short': 20,
    'sma_long': 50,
    'min_data_points': 50
}

# تنظیمات ریسک
RISK_CONFIG = {
    'max_position_size': 0.3,  # حداکثر 30% سرمایه در یک پوزیشن
    'stop_loss': 0.15,         # 15% استاپ لاس
    'take_profit': 0.25,       # 25% تیک پروفیت
    'risk_per_trade': 0.02     # 2% ریسک در هر معامله
}

# تنظیمات وب‌سوکت
WEBSOCKET_CONFIG = {
    'realtime_file': 'shared/realtime_prices.json',
    'update_interval': 30,     # ثانیه
    'max_retry': 5
}
