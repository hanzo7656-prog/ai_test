# database_manager.py - مدیریت پایگاه داده ساده برای تست
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class TradingDatabase:
    """پایگاه داده ساده برای داده‌های تاریخی"""
    
    def __init__(self):
        self.data_dir = "./trading_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """دریافت داده‌های تاریخی - نسخه ساده برای تست"""
        try:
            logger.info(f"📥 دریافت داده‌های تاریخی {symbol} برای {days} روز")
            
            # اگر فایل ذخیره شده وجود دارد، از آن استفاده کن
            file_path = os.path.join(self.data_dir, f"{symbol}_historical.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                if len(df) >= days:
                    return df.tail(days)
            
            # در غیر این صورت داده نمونه تولید کن
            return self._generate_sample_data(symbol, days)
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های {symbol}: {e}")
            return self._generate_sample_data(symbol, days)
    
    def _generate_sample_data(self, symbol: str, days: int) -> pd.DataFrame:
        """تولید داده نمونه واقعی‌تر"""
        np.random.seed(hash(symbol) % 1000)  # seed بر اساس سیمبل
        
        dates = pd.date_range(
            end=pd.Timestamp.now(), 
            periods=days, 
            freq='D'
        )
        
        # قیمت پایه بر اساس سیمبل
        base_prices = {
            'bitcoin': 45000,
            'ethereum': 2500, 
            'solana': 100,
            'binance-coin': 300,
            'cardano': 0.5,
            'ripple': 0.6,
            'default': 100
        }
        
        base_price = base_prices.get(symbol.lower(), base_prices['default'])
        prices = [base_price]
        
        for i in range(1, days):
            # تغییرات واقعی‌تر با روند
            volatility = 0.02 + (hash(symbol) % 10) / 100  # نوسان متفاوت
            trend = np.sin(i / 30) * 0.001  # روند سینوسی
            
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.1))  # جلوگیری از قیمت منفی
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [abs(np.random.normal(1000000, 500000)) for _ in range(days)]
        })
        
        df.set_index('timestamp', inplace=True)
        
        # ذخیره برای استفاده بعدی
        file_path = os.path.join(self.data_dir, f"{symbol}_historical.csv")
        df.to_csv(file_path)
        
        logger.info(f"✅ داده نمونه برای {symbol} تولید شد: {len(df)} رکورد")
        return df
    
    def save_market_data(self, symbol: str, data: Dict):
        """ذخیره داده‌های بازار"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_market.json")
            
            import json
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره داده‌های {symbol}: {e}")
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """دریافت داده‌های بازار ذخیره شده"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_market.json")
            
            if os.path.exists(file_path):
                import json
                with open(file_path, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"❌ خطا در خواندن داده‌های {symbol}: {e}")
        
        return None

# ایجاد نمونه گلوبال
trading_db = TradingDatabase()
