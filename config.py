# config.py - نسخه سازگار با Render
import os
from typing import List, Dict, Any

# ==================== تنظیمات اصلی API ====================
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': os.environ.get('COINSTATS_API_KEY', 'oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA='),
    'timeout': 30,
    'retry_attempts': 3,
    'rate_limit_per_minute': 60
}

# ==================== تنظیمات داده‌های خام ====================
RAW_DATA_CONFIG = {
    'save_raw_responses': True,
    'raw_data_folder': './raw_data',
    'compress_data': False,
    'keep_original_format': True
}

# ==================== مسیرهای داده ====================
RAW_DATA_PATHS = [
    "./raw_data",
    "../my-dataset/raw_data", 
    "./dataset/raw_data"
]

# ==================== تایم‌فریم‌های پشتیبانی شده ====================
SUPPORTED_TIMEFRAMES = [
    "1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"
]

# ==================== کوین‌های اصلی ====================
MAJOR_COINS = [
    "bitcoin", "ethereum", "solana", "binance-coin",
    "cardano", "ripple", "polkadot", "dogecoin", 
    "chainlink", "polygon", "avalanche", "litecoin"
]

# ==================== جفت ارزهای اصلی ====================
MAJOR_TRADING_PAIRS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt",
    "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
    "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
]

# ==================== تنظیمات WebSocket ====================
WEBSOCKET_CONFIG = {
    'url': 'wss://www.lbank.net/ws/V2/',
    'reconnect_delay': 5,
    'timeout': 10
}

# ==================== تنظیمات مدل AI ====================
AI_MODEL_CONFIG = {
    'sequence_length': 64,
    'feature_dim': 20,
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 3,
    'dropout': 0.1
}

# ==================== تنظیمات سرور ====================
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 8000)),
    'debug': os.environ.get('DEBUG', 'False').lower() == 'true',
    'workers': 1
}

# ==================== تنظیمات لاگ ====================
LOGGING_CONFIG = {
    'level': os.environ.get('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# ==================== اندپوینت‌های اصلی ====================
ENDPOINTS_CONFIG = {
    'coins': '/coins',
    'coins_charts': '/coins/charts',
    'coin_details': '/coins/{}',
    'coin_charts': '/coins/{}/charts',
    'price_avg': '/coins/price/avg',
    'exchange_price': '/coins/price/exchange',
    'tickers_exchanges': '/tickers/exchanges',
    'tickers_markets': '/tickers/markets',
    'fiats': '/fiats',
    'markets': '/markets',
    'currencies': '/currencies',
    'news_sources': '/news/sources',
    'news': '/news',
    'news_by_type': '/news/type/{}',
    'news_by_id': '/news/{}',
    'btc_dominance': '/insights/btc-dominance',
    'fear_greed': '/insights/fear-and-greed',
    'fear_greed_chart': '/insights/fear-and-greed/chart',
    'rainbow_chart': '/insights/rainbow-chart/{}'
}
