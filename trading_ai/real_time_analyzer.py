# real_time_analyzer.py - تحلیل بازار با داده‌های واقعی
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from technical_engine_complete import CompleteTechnicalEngine

from complete_coinstats_manager import coin_stats_manager
from database_manager import trading_db

logger = logging.getLogger(__name__)

class RealTimeMarketAnalyzer:
    """تحلیل‌گر بازار با داده‌های واقعی"""
    
    def __init__(self):
        self.tech_engine = CompleteTechnicalEngine()
        self.symbols = ['bitcoin', 'ethereum', 'solana', 'binance-coin']
        
    def fetch_real_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """دریافت داده‌های واقعی از CoinStats"""
        try:
            # دریافت داده‌های تاریخی از API
            chart_data = coin_stats_manager.get_coin_charts(symbol, period)
            
            if not chart_data or 'result' not in chart_data:
                logger.error(f"❌ داده‌های {symbol} دریافت نشد")
                return pd.DataFrame()
            
            # تبدیل به DataFrame
            records = []
            for item in chart_data['result']:
                records.append({
                    'timestamp': datetime.fromtimestamp(item.get('t', 0)),
                    'open': float(item.get('o', 0)),
                    'high': float(item.get('h', 0)),
                    'low': float(item.get('l', 0)),
                    'close': float(item.get('c', 0)),
                    'volume': float(item.get('v', 0))
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                # ذخیره در پایگاه داده
                trading_db.save_price_data(symbol, records)
            
            logger.info(f"✅ داده‌های واقعی {symbol} دریافت شد: {len(df)} رکورد")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_real_technical_indicators(self, symbol: str) -> Dict:
        """محاسبه اندیکاتورهای تکنیکال واقعی"""
        try:
            # دریافت داده‌های تاریخی
            df = trading_db.get_historical_data(symbol, 100)
            
            if df.empty:
                df = self.fetch_real_market_data(symbol, "100d")
            
            if df.empty or len(df) < 50:
                return {}
            
            # محاسبه اندیکاتورها
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            # استفاده از موتور تکنیکال موجود
            ohlc_data = {
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'close': df['close'].tolist(),
                'volume': df['volume'].tolist()
            }
            
            indicators = self.tech_engine.calculate_all_indicators(ohlc_data)
            
            # ذخیره اندیکاتورها
            indicator_data = {
                'timestamp': datetime.now().isoformat(),
                'rsi': indicators.get('rsi'),
                'macd': indicators.get('macd'),
                'bollinger_upper': indicators.get('bb_upper'),
                'bollinger_middle': indicators.get('sma_20'),  # SMA 20 به عنوان خط میانی
                'bollinger_lower': indicators.get('bb_lower'),
                'sma_20': indicators.get('sma_20'),
                'ema_12': indicators.get('ema_12'),
                'atr': self._calculate_atr(highs, lows, closes)
            }
            
            trading_db.save_technical_indicators(symbol, indicator_data)
            
            logger.info(f"✅ اندیکاتورهای تکنیکال {symbol} محاسبه شد")
            return indicator_data
            
        except Exception as e:
            logger.error(f"❌ خطا در محاسبه اندیکاتورهای {symbol}: {e}")
            return {}
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                      period: int = 14) -> float:
        """محاسبه Average True Range"""
        try:
            if len(highs) < period + 1:
                return 0.0
            
            tr = np.zeros(len(highs))
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr[i] = max(tr1, tr2, tr3)
            
            atr = np.mean(tr[-period:])
            return float(atr)
            
        except Exception as e:
            logger.error(f"❌ خطا در محاسبه ATR: {e}")
            return 0.0
    
    def get_current_market_sentiment(self) -> Dict:
        """دریافت احساسات بازار واقعی"""
        try:
            # دریافت شاخص ترس و طمع
            fear_greed_data = coin_stats_manager.get_fear_greed()
            
            # دریافت اخبار
            news_data = coin_stats_manager.get_news(limit=10)
            
            sentiment_score = 50  # پیش‌فرض
            
            if fear_greed_data and 'now' in fear_greed_data:
                sentiment_score = fear_greed_data['now'].get('value', 50)
            
            return {
                'fear_greed_index': sentiment_score,
                'news_count': len(news_data.get('result', [])),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت احساسات بازار: {e}")
            return {'fear_greed_index': 50, 'news_count': 0}
    
    def prepare_ai_training_data(self, symbol: str) -> pd.DataFrame:
        """آماده‌سازی داده‌های آموزشی برای AI"""
        df = trading_db.get_historical_data(symbol, 365)
        
        if df.empty:
            return pd.DataFrame()
        
        # ایجاد ویژگی‌های اضافی
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # ایجاد برچسب‌های هدف (سیگنال آینده)
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1  # بازده 5 روز آینده
        df['target'] = np.where(df['future_return'] > 0.02, 1, 
                               np.where(df['future_return'] < -0.02, -1, 0))
        
        # حذف مقادیر NaN
        df = df.dropna()
        
        return df

# ایجاد نمونه گلوبال
market_analyzer = RealTimeMarketAnalyzer()
