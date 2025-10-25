# config.py - نسخه بهینه شده
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# تنظیمات API - با کلید مستقیم از مستندات
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': 'oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=',  # کلید مستقیم از مستندات شما
    'timeout': 30,
    'retry_attempts': 3,
    'rate_limit_per_minute': 60,
    
    # اندپوینت‌های اصلی بر اساس مستندات
    'endpoints': {
        'coins_list': '/coins',
        'coins_charts': '/coins/charts',
        'coin_details': '/coins/{coin_id}',
        'coin_charts': '/coins/{coin_id}/charts',
        'price_avg': '/coins/price/avg',
        'exchange_price': '/coins/price/exchange',
        'tickers_exchanges': '/tickers/exchanges',
        'tickers_markets': '/tickers/markets',
        'fiats': '/fiats',
        'markets': '/markets',
        'currencies': '/currencies',
        'news_sources': '/news/sources',
        'news': '/news',
        'news_handpicked': '/news/type/handpicked',
        'news_detail': '/news/{news_id}',
        'btc_dominance': '/insights/btc-dominance',
        'fear_greed': '/insights/fear-and-greed',
        'fear_greed_chart': '/insights/fear-and-greed/chart',
        'rainbow_chart': '/insights/rainbow-chart/{coin_id}'
    }
}

# تنظیمات مسیرهای داده
DATA_CONFIG = {
    'raw_data_paths': [
        "./raw_data",
        "../my-dataset/raw_data", 
        "./dataset/raw_data"
    ],
    'processed_data_path': "./processed_data",
    'cache_path': "./cache"
}

# تایم‌فریم‌های پشتیبانی شده از مستندات
TIME_FRAME_CONFIG = {
    'supported_timeframes': [
        "1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"
    ],
    'default_timeframe': "1d",
    'chart_periods': {
        'hourly': '1h',
        'daily': '1d', 
        'weekly': '7d',
        'monthly': '1m',
        'yearly': '1y',
        'all': 'all'
    }
}

# کوین‌های اصلی از مستندات
COIN_CONFIG = {
    'major_coins': [
        "bitcoin", "ethereum", "solana", "binance-coin",
        "cardano", "ripple", "polkadot", "dogecoin", 
        "chainlink", "polygon", "avalanche", "litecoin"
    ],
    'default_currency': 'USD',
    'supported_currencies': ['USD', 'EUR', 'GBP', 'JPY', 'KRW']
}

# جفت ارزهای اصلی برای WebSocket
TRADING_CONFIG = {
    'major_trading_pairs': [
        "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt",
        "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
        "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
    ],
    'major_exchanges': [
        "Binance", "Coinbase", "Kraken", "Huobi", 
        "OKX", "KuCoin", "Gate.io", "Bitfinex"
    ]
}

# تنظیمات WebSocket
WEBSOCKET_CONFIG = {
    'url': 'wss://www.lbank.net/ws/V2/',
    'reconnect_delay': 5,
    'timeout': 10,
    'heartbeat_interval': 30
}

# تنظیمات مدل AI
AI_MODEL_CONFIG = {
    'sequence_length': 64,
    'feature_dim': 20,
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# تنظیمات سرور
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 1,
    'cors_origins': ['*']
}

# تنظیمات لاگ
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'file_path': './logs/app.log'
}

# تنظیمات فیلتر از مستندات
FILTER_CONFIG = {
    'sort_fields': [
        'rank', 'marketCap', 'price', 'volume', 
        'priceChange1h', 'priceChange1d', 'priceChange7d', 
        'name', 'symbol'
    ],
    'sort_directions': ['asc', 'desc'],
    'blockchains': [
        'ethereum', 'solana', 'binance-smart-chain', 
        'polygon', 'avalanche', 'bitcoin'
    ],
    'categories': [
        'defi', 'memecoins', 'gaming', 'nft', 
        'metaverse', 'layer-1', 'layer-2'
    ]
}

# تنظیمات کش
CACHE_CONFIG = {
    'ttl': 300,  # 5 minutes
    'max_size': 1000,
    'cleanup_interval': 600
}

def get_api_key():
    """دریافت کلید API از محیط یا کانفیگ"""
    return os.getenv('COINSTATS_API_KEY', API_CONFIG['api_key'])

def get_base_url():
    """دریافت آدرس پایه API"""
    return API_CONFIG['base_url']

def get_endpoint(endpoint_name, **kwargs):
    """دریافت آدرس کامل اندپوینت"""
    endpoint_template = API_CONFIG['endpoints'].get(endpoint_name)
    if endpoint_template and kwargs:
        return endpoint_template.format(**kwargs)
    return endpoint_template
