# advanced_technical_engine.py - نسخه بدون TA-Lib با داده‌های خام

import pandas as pd
import numpy as np
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
            self.indicators = ['RSI', 'MACD', 'BBANDS', 'STOCH', 'ATR', 'OBV', 'SMA', 'EMA']

class AdvancedTechnicalEngine:
    """موتور تحلیل تکنیکال پیشرفته بدون وابستگی به TA-Lib"""
    
    def __init__(self, config: TechnicalConfig = None):
        self.config = config or TechnicalConfig()
        self.feature_scalers = {}
        logger.info("🚀 Advanced Technical Engine Initialized - Raw Data Mode")

    def prepare_training_data(self, symbol: str, lookback_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی داده‌های آموزشی برای معماری اسپارس با داده‌های خام"""
        try:
            # دریافت داده‌های تاریخی خام
            df = self.get_historical_data(symbol, lookback_days)
            if df.empty:
                return None, None

            # محاسبه اندیکاتورها از داده خام
            df = self.calculate_all_indicators(df)

            # ایجاد دنباله‌های زمانی برای LSTM
            sequences, labels = self.create_sequences(df)

            logger.info(f"✅ داده‌های آموزشی {symbol} آماده شد: {len(sequences)} دنباله")
            return sequences, labels

        except Exception as e:
            logger.error(f"❌ خطا در آماده‌سازی داده‌های {symbol}: {e}")
            return None, None

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ایجاد دنباله‌های 60 کندلی برای آموزش"""
        sequences = []
        labels = []
        
        # استفاده از داده‌های خام اصلی
        features = ['open', 'high', 'low', 'close', 'volume']
        
        for i in range(self.config.sequence_length, len(df) - 5):
            # دنباله ورودی: 60 کندل اخیر از داده خام
            sequence = df[features].iloc[i-self.config.sequence_length:i].values
            sequences.append(sequence)

            # برچسب: تغییرات 5 کندل آینده
            future_prices = df['close'].iloc[i:i+5].values
            price_change = (future_prices[-1] / future_prices[0] - 1) * 100

            # کدگذاری برچسب برای طبقه بندی
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
        """محاسبه تمام اندیکاتورهای مورد نیاز برای آموزش - بدون TA-Lib"""
        # استخراج داده‌های خام
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        try:
            # اندیکاتورهای مومنتوم
            df['rsi_14'] = self._calculate_rsi(closes, 14)
            df['rsi_21'] = self._calculate_rsi(closes, 21)

            # MACD
            macd, macd_signal = self._calculate_macd(closes)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd - macd_signal

            # باندهای بولینگر
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # استوکاستیک
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            # حجم
            df['obv'] = self._calculate_obv(closes, volumes)

            # نوسان
            df['atr'] = self._calculate_atr(highs, lows, closes, 14)

            # میانگین‌های متحرک
            df['sma_20'] = self._calculate_sma(closes, 20)
            df['ema_12'] = self._calculate_ema(closes, 12)
            df['ema_26'] = self._calculate_ema(closes, 26)

            logger.info("✅ تمام اندیکاتورها با موتور داخلی محاسبه شد")

        except Exception as e:
            logger.error(f"❌ خطا در محاسبه اندیکاتورها: {e}")
            df = self._calculate_fallback_indicators(df)

        # محاسبه تغییرات
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        # پر کردن مقادیر NaN
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """محاسبه RSI از داده‌های خام"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')

        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # padding برای هم‌اندازی با طول اصلی
        rsi_padded = np.concatenate([np.full(period, 50), rsi])
        return rsi_padded[:len(prices)]

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """محاسبه MACD از داده‌های خام"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._calculate_ema(macd, signal)
        return macd, macd_signal

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """محاسبه باندهای بولینگر از داده‌های خام"""
        sma = self._calculate_sma(prices, period)
        rolling_std = pd.Series(prices).rolling(period).std().values
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band

    def _calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """محاسبه استوکاستیک از داده‌های خام"""
        stoch_k = np.zeros_like(closes)
        stoch_d = np.zeros_like(closes)

        for i in range(period, len(closes)):
            high_period = highs[i-period:i]
            low_period = lows[i-period:i]
            close_current = closes[i]
            highest_high = np.max(high_period)
            lowest_low = np.min(low_period)
            
            if highest_high != lowest_low:
                stoch_k[i] = 100 * (close_current - lowest_low) / (highest_high - lowest_low)
            else:
                stoch_k[i] = 50

        # محاسبه میانگین متحرک stoch_k
        stoch_d = pd.Series(stoch_k).rolling(3).mean().values
        return stoch_k, stoch_d

    def _calculate_obv(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """محاسبه On Balance Volume از داده‌های خام"""
        obv = np.zeros_like(closes)
        obv[0] = volumes[0]
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
                
        return obv

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """محاسبه Average True Range از داده‌های خام"""
        tr = np.zeros_like(highs)
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr[i] = max(tr1, tr2, tr3)
            
        atr = pd.Series(tr).rolling(period).mean().values
        return atr

    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """محاسبه Simple Moving Average از داده‌های خام"""
        return pd.Series(prices).rolling(period).mean().values

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """محاسبه Exponential Moving Average از داده‌های خام"""
        return pd.Series(prices).ewm(span=period).mean().values

    def _calculate_fallback_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای ساده در صورت خطا"""
        closes = df['close'].values
        
        # اندیکاتورهای پایه
        df['sma_20'] = self._calculate_sma(closes, 20)
        df['ema_12'] = self._calculate_ema(closes, 12)
        df['price_change'] = df['close'].pct_change()
        
        # مقادیر پیش‌فرض برای سایر اندیکاتورها
        df['rsi_14'] = 50
        df['macd'] = 0
        df['macd_signal'] = 0
        df['bb_upper'] = closes
        df['bb_lower'] = closes
        df['stoch_k'] = 50
        df['stoch_d'] = 50
        df['obv'] = df['volume'].cumsum()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        
        logger.info("⚠️ استفاده از اندیکاتورهای ساده به دلیل خطا")
        return df

    def get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """دریافت داده‌های تاریخی از دیتابیس با داده‌های خام"""
        try:
            from trading_ai.database_manager import trading_db
            df = trading_db.get_historical_data(symbol, days)
            
            if df.empty:
                df = self._generate_sample_data(days)
                
            return df
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های {symbol}: {e}")
            return self._generate_sample_data(days)

    def _generate_sample_data(self, days: int) -> pd.DataFrame:
        """تولید داده نمونه برای تست با داده‌های خام"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        np.random.seed(42)
        
        prices = [100]
        for i in range(1, days):
            change = np.random.normal(0.001, 0.02)
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
        """استخراج ویژگی‌های تکنیکال برای شبکه اسپارس از داده‌های خام"""
        features = []
        
        # ویژگی‌های قیمتی خام
        features.extend([
            df['close'].iloc[-1],
            df['high'].iloc[-1] - df['low'].iloc[-1],  # range
            df['volume'].iloc[-1],
        ])
        
        # ویژگی‌های اندیکاتور از داده خام
        if 'rsi_14' in df.columns:
            features.extend([
                df['rsi_14'].iloc[-1],
                df['macd'].iloc[-1],
                df['bb_width'].iloc[-1] if not np.isnan(df['bb_width'].iloc[-1]) else 0,
                df['stoch_k'].iloc[-1],
                df['atr'].iloc[-1] if not np.isnan(df['atr'].iloc[-1]) else 0
            ])
        
        # اگر ویژگی‌ها کم هستند، مقادیر پیش‌فرض اضافه کن
        while len(features) < self.config.feature_count:
            features.append(0.0)
            
        return np.array(features, dtype=np.float32)

    def get_raw_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """بررسی کیفیت داده‌های خام"""
        quality_metrics = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().sum(),
            'zero_volumes': (df['volume'] == 0).sum(),
            'price_consistency': self._check_price_consistency(df),
            'data_freshness': df.index.max() if hasattr(df.index, 'max') else None
        }
        
        # محاسبه نمره کیفیت
        quality_score = 0.0
        if quality_metrics['total_rows'] > 0:
            quality_score += 0.4  # 40% برای وجود داده
            
        if quality_metrics['missing_values'] == 0:
            quality_score += 0.3  # 30% برای عدم وجود مقادیر缺失
            
        if quality_metrics['zero_volumes'] == 0:
            quality_score += 0.3  # 30% برای حجم معاملات معتبر
            
        quality_metrics['quality_score'] = round(quality_score, 3)
        quality_metrics['quality_level'] = 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.5 else 'low'
        
        return quality_metrics

    def _check_price_consistency(self, df: pd.DataFrame) -> bool:
        """بررسی سازگاری قیمت‌ها"""
        try:
            # بررسی اینکه high >= low و high >= close >= low
            high_low_ok = (df['high'] >= df['low']).all()
            close_range_ok = ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all()
            return high_low_ok and close_range_ok
        except:
            return False

# ایجاد نمونه گلوبال
technical_engine = AdvancedTechnicalEngine()
