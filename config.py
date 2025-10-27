# config.py - نسخه کامل و اصلاح شده
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# 🔽 این خط مهم را اضافه کنید - مشکل WebSocket حل می‌شود
MAJOR_TRADING_PAIRS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt",
    "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
    "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
]

# تنظیمات API - با کلید مستقیم از مستندات
API_CONFIG = {
    'base_url': 'https://openapiv1.coinstats.app',
    'api_key': 'oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=',
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
    'cache_path': "./cache",
    'coinstats_cache': "./coinstats_cache"
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
    'supported_currencies': ['USD', 'EUR', 'GBP', 'JPY', 'KRW'],
    'popular_symbols': ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'XRP', 'DOT', 'LTC']
}

# جفت ارزهای اصلی برای WebSocket - با متغیر بالایی هماهنگ
TRADING_CONFIG = {
    'major_trading_pairs': MAJOR_TRADING_PAIRS,  # استفاده از متغیر بالایی
    'major_exchanges': [
        "Binance", "Coinbase", "Kraken", "Huobi", 
        "OKX", "KuCoin", "Gate.io", "Bitfinex"
    ],
    'websocket_pairs': MAJOR_TRADING_PAIRS  # برای سازگاری بیشتر
}

# تنظیمات WebSocket
WEBSOCKET_CONFIG = {
    'url': 'wss://www.lbank.net/ws/V2/',
    'reconnect_delay': 5,
    'timeout': 10,
    'heartbeat_interval': 30,
    'ping_interval': 30,
    'ping_timeout': 10,
    'subscription_batch_size': 10
}

# تنظیمات مدل AI برای trading_ai
AI_MODEL_CONFIG = {
    'sequence_length': 60,
    'feature_dim': 5,
    'd_model': 64,
    'n_heads': 4,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    
    # تنظیمات معماری اسپارس
    'sparse_network': {
        'total_neurons': 2500,
        'connections_per_neuron': 50,
        'specialty_groups': {
            "support_resistance": 800,
            "trend_detection": 700,
            "pattern_recognition": 600,
            "volume_analysis": 400
        },
        'hidden_dim': 128,
        'num_layers': 4,
        'time_steps': 6,
        'num_heads': 8
    }
}

# تنظیمات سرور
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 1,
    'cors_origins': ['*'],
    'render_port': 10000  # پورت پیش‌فرض Render
}

# تنظیمات لاگ
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'file_path': './logs/app.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
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
    ],
    'price_ranges': {
        'low': (0, 1),
        'medium': (1, 100),
        'high': (100, float('inf'))
    }
}

# تنظیمات کش
CACHE_CONFIG = {
    'ttl': 300,  # 5 minutes
    'max_size': 1000,
    'cleanup_interval': 600,
    'coinstats_cache_duration': 300  # 5 minutes for CoinStats
}

# تنظیمات تحلیل تکنیکال
TECHNICAL_ANALYSIS_CONFIG = {
    'indicators': ['RSI', 'MACD', 'BBANDS', 'STOCH', 'ATR', 'OBV', 'SMA', 'EMA'],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'stoch_period': 14,
    'atr_period': 14,
    'sma_periods': [5, 10, 20, 50, 200],
    'ema_periods': [12, 26, 50]
}

# تنظیمات AI Analysis
AI_ANALYSIS_CONFIG = {
    'supported_periods': ["1h", "4h", "1d", "7d", "30d", "90d", "all"],
    'analysis_types': ["comprehensive", "technical", "sentiment", "momentum", "pattern"],
    'default_symbols': ['BTC', 'ETH', 'SOL', 'BNB'],
    'max_symbols_per_request': 5,
    'confidence_thresholds': {
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3
    }
}

# تنظیمات سیستم
SYSTEM_CONFIG = {
    'project_name': 'AI Trading Assistant',
    'version': '3.0.0',
    'environment': os.getenv('ENVIRONMENT', 'production'),
    'max_memory_mb': 512,
    'max_request_timeout': 30,
    'rate_limiting': {
        'max_requests_per_minute': 60,
        'max_requests_per_hour': 1000
    }
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

def get_port():
    """دریافت پورت از محیط"""
    return int(os.getenv('PORT', SERVER_CONFIG['port']))

def get_websocket_pairs():
    """دریافت جفت ارزهای WebSocket"""
    return MAJOR_TRADING_PAIRS

def is_debug():
    """بررسی حالت دیباگ"""
    return os.getenv('DEBUG', 'false').lower() == 'true'

def get_cache_duration():
    """دریافت مدت زمان کش"""
    return CACHE_CONFIG['ttl']

# تنظیمات پیش‌فرض برای تست
if __name__ == "__main__":
    print("✅ Config loaded successfully")
    print(f"🔑 API Key: {get_api_key()[:10]}...")
    print(f"🌐 Base URL: {get_base_url()}")
    print(f"📊 Trading Pairs: {len(MAJOR_TRADING_PAIRS)} pairs")
    print(f"🧠 AI Model: {AI_MODEL_CONFIG['sparse_network']['total_neurons']} neurons")
    print(f"🚀 Server Port: {get_port()}")
