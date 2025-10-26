import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TechnicalConfig:
    """تنظیمات موتور تکنیکال برای معماری اسپارس"""
    sequence_length: int = 60
    feature_count: int = 5
    indicators: List[str] = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = ['RSI', 'MACD', 'BBANDS', 'STOCH', 'ATR', 'OBV']

class AdvancedTechnicalEngine:
    """موتور تحلیل تکنیکال پیشرفته برای آموزش معماری اسپارس"""
    
    def __init__(self, config: TechnicalConfig = None):
        self.config = config or TechnicalConfig()
        self.feature_scalers = {}
        
    def prepare_training_data(self, symbol: str, lookback_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی داده‌های آموزشی برای معماری اسپارس"""
        try:
            # دریافت داده‌های تاریخی
            df = self.get_historical_data(symbol, lookback_days)
            if df.empty:
                return None, None
            
            # محاسبه اندیکاتورها
            df = self.calculate_all_indicators(df)
            
            # ایجاد دنباله‌های زمانی برای LSTM
            sequences, labels = self.create_sequences(df)
            
            logger.info(f"✅ داده‌های آموزشی {symbol} آماده شد: {len(sequences)} دنباله")
            return sequences, labels
            
        except Exception as e:
            logger.error(f"❌ خطا در آماده‌سازی داده‌های {symbol}: {e}")
            return None, None
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ایجاد دنباله‌های 60 کندلی برای آموزش LSTM"""
        sequences = []
        labels = []
        
        features = ['open', 'high', 'low', 'close', 'volume']
        
        for i in range(self.config.sequence_length, len(df) - 5):
            # دنباله ورودی: 60 کندل اخیر
            sequence = df[features].iloc[i-self.config.sequence_length:i].values
            sequences.append(sequence)
            
            # برچسب: تغییرات 5 کندل آینده
            future_prices = df['close'].iloc[i:i+5].values
            price_change = (future_prices[-1] / future_prices[0] - 1) * 100
            
            # کدگذاری برچسب برای طبقه‌بندی
            if price_change > 2:
                label = 0  # صعودی قوی
            elif price_change > 0.5:
                label = 1  # صعودی ضعیف
            elif price_change < -2:
                label = 2  # نزولی قوی
            elif price_change < -0.5:
                label = 3  # نزولی ضعیف
            else:
                label = 4  # خنثی
                
            labels.append(label)
        
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه تمام اندیکاتورهای مورد نیاز برای آموزش"""
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        # اندیکاتورهای مومنتوم
        df['rsi_14'] = talib.RSI(closes, timeperiod=14)
        df['rsi_21'] = talib.RSI(closes, timeperiod=21)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(closes)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # باندهای بولینگر
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # استوکاستیک
        slowk, slowd = talib.STOCH(highs, lows, closes)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # حجم
        df['obv'] = talib.OBV(closes, volumes)
        
        # نوسان
        df['atr'] = talib.ATR(highs, lows, closes, timeperiod=14)
        
        # میانگین‌های متحرک
        df['sma_20'] = talib.SMA(closes, timeperiod=20)
        df['ema_12'] = talib.EMA(closes, timeperiod=12)
        df['ema_26'] = talib.EMA(closes, timeperiod=26)
        
        # محاسبه تغییرات
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # پر کردن مقادیر NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """دریافت داده‌های تاریخی از دیتابیس"""
        # استفاده از دیتابیس موجود
        from database_manager import trading_db
        df = trading_db.get_historical_data(symbol, days)
        
        if df.empty:
            # شبیه‌سازی داده برای تست
            df = self._generate_sample_data(days)
            
        return df
    
    def _generate_sample_data(self, days: int) -> pd.DataFrame:
        """تولید داده نمونه برای تست (موقت)"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        np.random.seed(42)
        prices = [100]
        for i in range(1, days):
            change = np.random.normal(0.001, 0.02)  # تغییرات روزانه
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            'close': prices,
            'volume': [abs(np.random.normal(1000000, 200000)) for _ in range(days)]
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def extract_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """استخراج ویژگی‌های تکنیکال برای شبکه اسپارس"""
        features = []
        
        # ویژگی‌های قیمتی
        features.extend([
            df['close'].iloc[-1],
            df['high'].iloc[-1] - df['low'].iloc[-1],  # range
            df['volume'].iloc[-1],
        ])
        
        # ویژگی‌های اندیکاتوری
        if 'rsi_14' in df.columns:
            features.extend([
                df['rsi_14'].iloc[-1],
                df['macd'].iloc[-1],
                df['bb_width'].iloc[-1],
                df['stoch_k'].iloc[-1],
                df['atr'].iloc[-1]
            ])
        
        return np.array(features, dtype=np.float32)

# ایجاد نمونه گلوبال
technical_engine = AdvancedTechnicalEngine()
