# data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas_ta as ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """پردازش و پاک‌سازی داده‌های بازار با قابلیت‌های پیشرفته"""
    
    def __init__(self):
        self.processed_data = {}
        self.technical_indicators = {}
        
    def clean_coin_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """پاک‌سازی داده‌های خام کوین‌ها"""
        if not raw_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(raw_data)
            
            # تبدیل انواع داده
            numeric_columns = [
                'price', 'priceBtc', 'volume', 'marketCap', 
                'availableSupply', 'totalSupply', 'fullyDilutedValuation',
                'priceChange1h', 'priceChange1d', 'priceChange1w',
                'rank'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # حذف داده‌های نامعتبر
            df = df.dropna(subset=['price', 'marketCap'])
            
            # اضافه کردن محاسبات اضافی
            if 'price' in df.columns and 'volume' in df.columns:
                df['price_volume_ratio'] = df['price'] / df['volume']
                df['market_cap_rank_ratio'] = df['marketCap'] / df['rank']
            
            print(f"✅ داده‌های {len(df)} کوین پاک‌سازی شد")
            return df
            
        except Exception as e:
            print(f"❌ خطا در پاک‌سازی داده‌ها: {e}")
            return pd.DataFrame()
    
    def process_chart_data(self, chart_data: List[Dict]) -> pd.DataFrame:
        """پردازش داده‌های چارت با قابلیت‌های پیشرفته"""
        if not chart_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(chart_data)
            
            # تبدیل timestamp به datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('datetime').reset_index(drop=True)
            
            # اطمینان از numeric بودن ستون‌های عددی
            numeric_columns = ['price', 'volume', 'market_cap', 'high', 'low', 'open']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # اگر high/low/open وجود ندارند، از price استفاده کن
            if 'high' not in df.columns and 'price' in df.columns:
                df['high'] = df['price']
            if 'low' not in df.columns and 'price' in df.columns:
                df['low'] = df['price']
            if 'open' not in df.columns and 'price' in df.columns:
                df['open'] = df['price']
            
            # حذف داده‌های نامعتبر
            df = df.dropna(subset=['price']).reset_index(drop=True)
            
            print(f"✅ داده‌های چارت پردازش شد ({len(df)} نقطه داده)")
            return df
            
        except Exception as e:
            print(f"❌ خطا در پردازش داده‌های چارت: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای تکنیکال پیشرفته با pandas-ta"""
        if df.empty or 'price' not in df.columns:
            return df
        
        try:
            # ایجاد کپی از داده‌ها
            result_df = df.copy()
            
            # قیمت برای محاسبات
            price = result_df['price']
            high = result_df['high'] if 'high' in result_df.columns else price
            low = result_df['low'] if 'low' in result_df.columns else price
            volume = result_df['volume'] if 'volume' in result_df.columns else None
            
            # 1. اندیکاتورهای روند (Trend)
            result_df['sma_20'] = ta.sma(price, length=20)
            result_df['sma_50'] = ta.sma(price, length=50)
            result_df['sma_100'] = ta.sma(price, length=100)
            result_df['ema_12'] = ta.ema(price, length=12)
            result_df['ema_26'] = ta.ema(price, length=26)
            
            # 2. اندیکاتورهای مومنتوم (Momentum)
            result_df['rsi_14'] = ta.rsi(price, length=14)
            result_df['rsi_21'] = ta.rsi(price, length=21)
            
            # MACD
            macd = ta.macd(price, fast=12, slow=26, signal=9)
            if macd is not None:
                result_df['macd'] = macd['MACD_12_26_9']
                result_df['macd_signal'] = macd['MACDs_12_26_9']
                result_df['macd_histogram'] = macd['MACDh_12_26_9']
            
            # Stochastic
            stoch = ta.stoch(high, low, price, k=14, d=3)
            if stoch is not None:
                result_df['stoch_k'] = stoch['STOCHk_14_3_3']
                result_df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            # Williams %R
            result_df['williams_r'] = ta.willr(high, low, price, length=14)
            
            # 3. اندیکاتورهای نوسان (Volatility)
            # Bollinger Bands
            bb = ta.bbands(price, length=20, std=2)
            if bb is not None:
                result_df['bb_upper'] = bb['BBU_20_2.0']
                result_df['bb_middle'] = bb['BBM_20_2.0']
                result_df['bb_lower'] = bb['BBL_20_2.0']
                result_df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            
            # ATR (Average True Range)
            result_df['atr_14'] = ta.atr(high, low, price, length=14)
            
            # 4. اندیکاتورهای حجم (Volume)
            if volume is not None:
                result_df['volume_sma_20'] = ta.sma(volume, length=20)
                result_df['obv'] = ta.obv(price, volume)
                
                # VWAP (اگر داده‌های تایم‌فریم دقیقه‌ای داشته باشیم)
                if 'datetime' in result_df.columns:
                    try:
                        vwap = ta.vwap(high, low, price, volume, 
                                     anchor=result_df['datetime'].dt.time)
                        if vwap is not None:
                            result_df['vwap'] = vwap
                    except:
                        pass
            
            # 5. اندیکاتورهای سیگنال ترکیبی
            result_df = self._calculate_composite_signals(result_df)
            
            # 6. محاسبات آماری
            result_df = self._calculate_statistical_measures(result_df)
            
            print(f"✅ {len([col for col in result_df.columns if col not in df.columns])} اندیکاتور محاسبه شد")
            return result_df
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورها: {e}")
            # Fallback به محاسبات ساده
            return self._calculate_basic_indicators(df)
    
    def _calculate_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه سیگنال‌های ترکیبی"""
        try:
            # سیگنال روند
            if all(col in df.columns for col in ['sma_20', 'sma_50']):
                df['trend_strength'] = np.where(
                    df['sma_20'] > df['sma_50'], 
                    (df['sma_20'] - df['sma_50']) / df['sma_50'] * 100,
                    (df['sma_50'] - df['sma_20']) / df['sma_20'] * 100
                )
            
            # سیگنال RSI
            if 'rsi_14' in df.columns:
                df['rsi_signal'] = np.select([
                    df['rsi_14'] > 70,
                    df['rsi_14'] < 30,
                    (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70)
                ], ['اشباع خرید', 'اشباع فروش', 'خنثی'], default='نامشخص')
            
            # سیگنال MACD
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                df['macd_signal_cross'] = np.where(
                    (df['macd'] > df['macd_signal']) & 
                    (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
                    'خرید',
                    np.where(
                        (df['macd'] < df['macd_signal']) & 
                        (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                        'فروش',
                        'خنثی'
                    )
                )
            
            # سیگنال بولینگر
            if all(col in df.columns for col in ['price', 'bb_upper', 'bb_lower']):
                df['bb_signal'] = np.select([
                    df['price'] > df['bb_upper'],
                    df['price'] < df['bb_lower'],
                    (df['price'] >= df['bb_lower']) & (df['price'] <= df['bb_upper'])
                ], ['بالای باند', 'زیر باند', 'درون باند'], default='نامشخص')
            
            return df
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه سیگنال‌های ترکیبی: {e}")
            return df
    
    def _calculate_statistical_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه معیارهای آماری"""
        try:
            if 'price' not in df.columns:
                return df
            
            price = df['price']
            
            # نوسان
            df['volatility_20'] = price.rolling(window=20).std() / price.rolling(window=20).mean() * 100
            df['volatility_50'] = price.rolling(window=50).std() / price.rolling(window=50).mean() * 100
            
            # بازده روزانه
            df['daily_return'] = price.pct_change() * 100
            
            # کشیدگی و چولگی
            if len(price) >= 30:
                try:
                    df['returns_skewness_30'] = price.pct_change().rolling(window=30).apply(
                        lambda x: stats.skew(x.dropna()) if x.dropna().size > 0 else 0, raw=False
                    )
                    df['returns_kurtosis_30'] = price.pct_change().rolling(window=30).apply(
                        lambda x: stats.kurtosis(x.dropna()) if x.dropna().size > 0 else 0, raw=False
                    )
                except:
                    pass
            
            # همبستگی با زمان (برای تشخیص روند)
            if len(df) >= 10:
                try:
                    time_corr = df.reset_index().index.to_series().rolling(window=10).corr(price)
                    df['price_time_correlation'] = time_corr.values
                except:
                    pass
            
            return df
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه معیارهای آماری: {e}")
            return df
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبات پایه اندیکاتورها (fallback)"""
        try:
            result_df = df.copy()
            price = result_df['price']
            
            # میانگین‌های متحرک
            result_df['sma_20'] = price.rolling(window=20).mean()
            result_df['sma_50'] = price.rolling(window=50).mean()
            
            # RSI
            result_df['rsi_14'] = self.calculate_rsi(price, 14)
            
            # نوسان
            result_df['volatility_20'] = price.rolling(window=20).std() / price.rolling(window=20).mean() * 100
            
            print("✅ اندیکاتورهای پایه محاسبه شد (fallback)")
            return result_df
            
        except Exception as e:
            print(f"❌ خطا در محاسبات پایه: {e}")
            return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه RSI (fallback)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def generate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """تولید سیگنال‌های معاملاتی از داده‌های پردازش شده"""
        if df.empty:
            return {"error": "داده‌های ناکافی"}
        
        try:
            # آخرین داده
            latest = df.iloc[-1]
            
            signals = {
                "timestamp": datetime.now().isoformat(),
                "current_price": latest.get('price', 0),
                "technical_indicators": {},
                "trading_signals": {},
                "risk_metrics": {}
            }
            
            # اندیکاتورهای تکنیکال
            if 'rsi_14' in latest:
                signals["technical_indicators"]["rsi"] = {
                    "value": round(latest['rsi_14'], 2),
                    "signal": latest.get('rsi_signal', 'نامشخص'),
                    "interpretation": "قوی" if latest['rsi_14'] > 70 or latest['rsi_14'] < 30 else "متوسط"
                }
            
            if all(col in latest for col in ['macd', 'macd_signal']):
                signals["technical_indicators"]["macd"] = {
                    "value": round(latest['macd'], 4),
                    "signal": latest.get('macd_signal', 0),
                    "histogram": round(latest.get('macd_histogram', 0), 4),
                    "trend": "صعودی" if latest['macd'] > latest['macd_signal'] else "نزولی"
                }
            
            if all(col in latest for col in ['sma_20', 'sma_50']):
                signals["technical_indicators"]["moving_averages"] = {
                    "sma_20": round(latest['sma_20'], 2),
                    "sma_50": round(latest['sma_50'], 2),
                    "trend": "صعودی" if latest['sma_20'] > latest['sma_50'] else "نزولی",
                    "strength": abs(latest['sma_20'] - latest['sma_50']) / latest['sma_50'] * 100
                }
            
            # سیگنال‌های معاملاتی
            buy_signals = 0
            sell_signals = 0
            
            if latest.get('rsi_signal') == 'اشباع فروش':
                buy_signals += 1
            elif latest.get('rsi_signal') == 'اشباع خرید':
                sell_signals += 1
            
            if latest.get('macd_signal_cross') == 'خرید':
                buy_signals += 1
            elif latest.get('macd_signal_cross') == 'فروش':
                sell_signals += 1
            
            if latest.get('bb_signal') == 'زیر باند':
                buy_signals += 1
            elif latest.get('bb_signal') == 'بالای باند':
                sell_signals += 1
            
            # تصمیم نهایی
            if buy_signals > sell_signals:
                final_signal = "خرید"
                confidence = buy_signals / (buy_signals + sell_signals) * 100
            elif sell_signals > buy_signals:
                final_signal = "فروش"
                confidence = sell_signals / (buy_signals + sell_signals) * 100
            else:
                final_signal = "خنثی"
                confidence = 50
            
            signals["trading_signals"] = {
                "final_signal": final_signal,
                "confidence": round(confidence, 2),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "total_signals": buy_signals + sell_signals
            }
            
            # معیارهای ریسک
            if 'volatility_20' in latest:
                signals["risk_metrics"]["volatility"] = {
                    "value": round(latest['volatility_20'], 2),
                    "level": "بالا" if latest['volatility_20'] > 5 else "متوسط" if latest['volatility_20'] > 2 else "پایین"
                }
            
            if 'atr_14' in latest and 'price' in latest:
                atr_percentage = (latest['atr_14'] / latest['price']) * 100
                signals["risk_metrics"]["atr"] = {
                    "value": round(latest['atr_14'], 2),
                    "percentage": round(atr_percentage, 2),
                    "risk_level": "بالا" if atr_percentage > 3 else "متوسط" if atr_percentage > 1 else "پایین"
                }
            
            return signals
            
        except Exception as e:
            return {"error": f"خطا در تولید سیگنال‌ها: {e}"}
    
    def get_technical_summary(self, df: pd.DataFrame) -> Dict:
        """خلاصه تحلیل تکنیکال"""
        if df.empty:
            return {"error": "داده‌های ناکافی"}
        
        try:
            latest = df.iloc[-1]
            summary = {
                "price_action": {
                    "current_price": latest.get('price', 0),
                    "price_change_24h": latest.get('priceChange1d', 0),
                    "support_level": df['price'].min(),
                    "resistance_level": df['price'].max()
                },
                "momentum": {},
                "trend": {},
                "volatility": {},
                "overall_assessment": ""
            }
            
            # مومنتوم
            if 'rsi_14' in latest:
                summary["momentum"]["rsi"] = {
                    "value": round(latest['rsi_14'], 2),
                    "status": "اشباع خرید" if latest['rsi_14'] > 70 else "اشباع فروش" if latest['rsi_14'] < 30 else "عادی"
                }
            
            # روند
            if all(col in latest for col in ['sma_20', 'sma_50']):
                trend = "صعودی" if latest['sma_20'] > latest['sma_50'] else "نزولی"
                strength = abs(latest['sma_20'] - latest['sma_50']) / latest['sma_50'] * 100
                summary["trend"]["direction"] = trend
                summary["trend"]["strength"] = round(strength, 2)
            
            # نوسان
            if 'volatility_20' in latest:
                summary["volatility"]["level"] = round(latest['volatility_20'], 2)
                summary["volatility"]["assessment"] = "بالا" if latest['volatility_20'] > 5 else "متوسط" if latest['volatility_20'] > 2 else "پایین"
            
            # ارزیابی کلی
            buy_score = 0
            if latest.get('rsi_14', 50) < 40: buy_score += 1
            if latest.get('macd_signal_cross') == 'خرید': buy_score += 1
            if latest.get('bb_signal') == 'زیر باند': buy_score += 1
            
            if buy_score >= 2:
                summary["overall_assessment"] = "شرایط مطلوب برای خرید"
            elif buy_score >= 1:
                summary["overall_assessment"] = "احتیاط در خرید"
            else:
                summary["overall_assessment"] = "انتظار برای شرایط بهتر"
            
            return summary
            
        except Exception as e:
            return {"error": f"خطا در تولید خلاصه: {e}"}


# تست ماژول
if __name__ == "__main__":
    # تست کلاس
    processor = DataProcessor()
    
    # داده‌های نمونه
    sample_data = [
        {"timestamp": 1703000000, "price": 45000, "volume": 1000000},
        {"timestamp": 1703003600, "price": 45500, "volume": 1200000},
        {"timestamp": 1703007200, "price": 45200, "volume": 900000},
    ]
    
    processed = processor.process_chart_data(sample_data)
    print(f"داده‌های پردازش شده: {len(processed)} رکورد")
    
    if not processed.empty:
        with_indicators = processor.calculate_technical_indicators(processed)
        print(f"ستون‌های اندیکاتور: {[col for col in with_indicators.columns if col not in processed.columns]}")
        
        signals = processor.generate_trading_signals(with_indicators)
        print("سیگنال‌ها:", signals)
