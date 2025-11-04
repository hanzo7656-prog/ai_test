"""
تنظیمات سیستم کش GitHub DB
"""

import os
from typing import Dict, Any

# تنظیمات GitHub DB
GITHUB_DB_CONFIG = {
    'repo_path': './github_db_data',
    'compression_threshold_days': 7,
    'cleanup_threshold_days': 30,
    'max_live_files': 500,
    'backup_enabled': True
}

# تنظیمات اسکن دسته‌ای
BATCH_SCAN_CONFIG = {
    'batch_size': 25,
    'total_symbols': 500,
    'scan_interval_minutes': 5,
    'retry_attempts': 3,
    'timeout_seconds': 30
}

# تنظیمات فشرده‌سازی
COMPRESSION_CONFIG = {
    'enabled': True,
    'algorithm': 'gzip',
    'compression_level': 6,
    'auto_cleanup': True
}

# تنظیمات API یکپارچه‌سازی
API_INTEGRATION_CONFIG = {
    'coinstats_timeout': 20,
    'max_concurrent_requests': 5,
    'cache_ttl_minutes': 10
}

def get_github_db_path() -> str:
    """دریافت مسیر GitHub DB"""
    return os.getenv('GITHUB_DB_PATH', GITHUB_DB_CONFIG['repo_path'])

def get_batch_size() -> int:
    """دریافت سایز دسته"""
    return int(os.getenv('BATCH_SIZE', BATCH_SCAN_CONFIG['batch_size']))

def get_total_symbols() -> int:
    """دریافت تعداد کل ارزها"""
    return int(os.getenv('TOTAL_SYMBOLS', BATCH_SCAN_CONFIG['total_symbols']))
