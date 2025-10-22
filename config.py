# config.py
# تنظیمات پروژه

# مسیرهای داده
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

# جفت‌ارزهای اصلی برای WebSocket
MAJOR_TRADING_PAIRS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", 
    "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
    "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
]

# تنظیمات API
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': 'oYGllJrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=',
    'timeout': 30,
    'retry_attempts': 3
}
