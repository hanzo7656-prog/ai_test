# config.py - تنظیمات سیستم تریدینگ واقعی
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TradingConfig:
    """تنظیمات تریدینگ"""
    
    # نمادهای تحت پوشش
    SYMBOLS: List[str] = ['bitcoin', 'ethereum', 'solana', 'binance-coin']
    
    # تنظیمات مدل
    MODEL_HIDDEN_DIMS: List[int] = (256, 128, 64, 32)
    MODEL_DROPOUT: float = 0.2
    TRAINING_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    
    # تنظیمات ریسک
    MAX_POSITION_SIZE: float = 0.1  # 10% سرمایه
    STOP_LOSS_ATR_MULTIPLIER: float = 1.5
    TAKE_PROFIT_RATIO: float = 2.0
    
    # تنظیمات داده
    LOOKBACK_DAYS: int = 365
    SEQUENCE_LENGTH: int = 60
    
    # تنظیمات بک‌تست
    INITIAL_BALANCE: float = 10000
    COMMISSION_RATE: float = 0.001  # 0.1%

# ایجاد نمونه تنظیمات
trading_config = TradingConfig()
