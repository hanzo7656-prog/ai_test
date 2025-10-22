# 📁 sparse_config.py

# تنظیمات منابع داده
DATA_SOURCES = {
    "github_base_url": "https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data",
    "coinstats_api_url": "https://openapiv1.coinstats.app",
    "ai_service_url": "https://ai-test-2nxq.onrender.com"
}

# تنظیمات API Key (از environment variables خوانده شود)
import os
COINSTATS_API_KEY = os.getenv("COINSTATS_API_KEY", "oYGllJrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=")

# تنظیمات کش
CACHE_CONFIG = {
    "max_size": 200,
    "ttl": 300  # 5 minutes
}
