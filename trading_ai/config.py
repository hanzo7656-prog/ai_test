from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TradingConfig:
    """تنظیمات اصلی سیستم تحلیل تکنیکال"""
    
    # نمادهای تحت پوشش
    SYMBOLS: List[str] = ['bitcoin', 'ethereum', 'solana', 'binance-coin']
    
    # تنظیمات معماری اسپارس
    SPARSE_NEURONS: int = 2500
    SPARSE_CONNECTIONS: int = 50
    TEMPORAL_SEQUENCE: int = 60
    INPUT_FEATURES: int = 5
    
    # تنظیمات آموزش
    TRAINING_EPOCHS: int = 30
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    
    # تنظیمات تحلیل تکنیکال
    TECHNICAL_INDICATORS: List[str] = ['RSI', 'MACD', 'BBANDS', 'STOCH', 'ATR', 'OBV']
    LOOKBACK_DAYS: int = 365
    
    # تنظیمات اسکن
    SCAN_INTERVAL: int = 300  # ثانیه
    CONFIDENCE_THRESHOLD: float = 0.7

# ایجاد نمونه‌های پیکربندی
trading_config = TradingConfig()
technical_config = TechnicalConfig()
sparse_config = SparseConfig()
