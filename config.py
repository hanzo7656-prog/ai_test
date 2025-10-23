# config.py - نسخه بهینه شده
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# تنظیمات مسیرهای داده
RAW_DATA_PATHS = [
    "./raw_data",
    "../my-dataset/raw_data", 
    "./dataset/raw_data"
]

# تایم‌فریم‌های پشتیبانی شده
SUPPORTED_TIMEFRAMES = [
    "1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"
]

# کوین‌های اصلی
MAJOR_COINS = [
    "bitcoin", "ethereum", "solana", "binance-coin",
    "cardano", "ripple", "polkadot", "dogecoin", 
    "chainlink", "polygon", "avalanche", "litecoin"
]

# جفت ارزهای اصلی برای WebSocket
MAJOR_TRADING_PAIRS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt",
    "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
    "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
]

# تنظیمات API - امن شده
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': os.getenv('COINSTATS_API_KEY', ''),
    'timeout': 30,
    'retry_attempts': 3,
    'rate_limit_per_minute': 60
}

# تنظیمات WebSocket
WEBSOCKET_CONFIG = {
    'url': 'wss://www.lbank.net/ws/V2/',
    'reconnect_delay': 5,
    'timeout': 10
}

# تنظیمات مدل AI
AI_MODEL_CONFIG = {
    'sequence_length': 64,
    'feature_dim': 20,
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 3,
    'dropout': 0.1
}

# تنظیمات سرور
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 1
}

# تنظیمات لاگ
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}
