# data_processor_optimized.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas_ta as ta
from scipy import stats
import warnings
import gc
from sys import getsizeof
warnings.filterwarnings('ignore')

class OptimizedDataProcessor:
    """پردازشگر داده بهینه‌شده برای مصرف حافظه"""
    
    def __init__(self):
        self.processed_data = {}
        self.technical_indicators = {}
        self.processing_count = 0
        
        # تنظیمات بهینه‌سازی
        try:
            from config import MEMORY_OPTIMIZATION, INDICATOR_CONFIG
            self.mem_config = MEMORY_OPTIMIZATION
            self.indicator_config = INDICATOR_CONFIG
        except ImportError:
            self.mem_config = {
                'dtype_optimization': True,
                'downcast_numbers': True,
                'remove_unused_columns': True,
                'chunk_processing': True,
                'chunk_size': 1000,
                'max_data_points': 5000,
                'cleanup_interval': 100
            }
            self.indicator_config = {
                'enable_basic_indicators': True,
                'enable_advanced_indicators': False,
                'rsi_period': 14,
                'sma_periods': [20, 50],
                'max_indicators': 10
            }
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """بهینه‌سازی مصرف حافظه دیتافریم"""
        if df.empty:
            return df
            
        try:
            # کپی نکن - روی همان object کار کن
            result_df = df
            
            # 1. محدود کردن تعداد سطرها
            max_points = self.mem_config.get('max_data_points', 5000)
            if len(result_df) > max_points:
                result_df = result_df.iloc[-max_points:].reset_index(drop=True)
            
            # 2. بهینه‌سازی انواع داده‌های عددی
            if self.mem_config.get('dtype_optimization', True):
                numeric_columns = result_df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                    # downcast اعداد
                    if self.mem_config.get('downcast_numbers', True):
                        result_df[col] = pd.to_numeric(
                            result_df[col], 
                            downcast='float' if result_df[col].dtype == 'float64' else 'integer'
                        )
            
            # 3. حذف ستون‌های غیرضروری
            if self.mem_config.get('remove_unused_columns', True):
                essential_columns = ['timestamp', 'datetime', 'price', 'volume', 'high', 'low', 'open']
                existing_essential = [col for col in essential_columns if col in result_df.columns]
                extra_columns = [col for col in result_df.columns if col not in existing_essential]
                
                # فقط 5 ستون اضافی نگه دار
                if len(extra_columns) > 5:
                    columns_to_keep = existing_essential + extra_columns[:5]
                    result_df = result_df[columns_to_keep]
            
            return result_df
            
        except Exception as e:
            print(f"⚠️ خطا در بهینه‌سازی حافظه: {e}")
            return df
    
    def process_chart_data_optimized(self, chart_data: List[Dict]) -> pd.DataFrame:
        """پردازش داده‌های چارت با بهینه‌سازی حافظه"""
        if not chart_data:
            return pd.DataFrame()
        
        try:
            # پردازش chunk-based برای داده‌های بزرگ
            if len(chart_data) > self.mem_config.get('chunk_size', 1000) and \
               self.mem_config.get('chunk_processing', True):
                return self._process_large_data_in_chunks(chart_data)
            
            # پردازش عادی برای داده‌های کوچک
            df = pd.DataFrame(chart_data)
            df = self.optimize_dataframe_memory(df)
            
            # تبدیل timestamp به datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('datetime').reset_index(drop=True)
            
            # اطمینان از numeric بودن ستون‌های اصلی
            essential_numeric = ['price', 'volume', 'high', 'low', 'open']
            for col in essential_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # تکمیل داده‌های缺失 فقط برای ستون‌های ضروری
            if 'high' not in df.columns and 'price' in df.columns:
                df['high'] = df['price']
            if 'low' not in df.columns and 'price' in df.columns:
                df['low'] = df['price']
            if 'open' not in df.columns and 'price' in df.columns:
                df['open'] = df['price']
            
            # حذف داده‌های نامعتبر فقط برای قیمت
            df = df.dropna(subset=['price']).reset_index(drop=True)
            
            # پاک‌سازی حافظه
            self._cleanup_memory()
            
            print(f"✅ داده‌های چارت پردازش شد ({len(df)} نقطه داده) - حافظه: {self._get_memory_usage()}") 
            return df
            
        except Exception as e:
            print(f"❌ خطا در پردازش داده‌های چارت: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورها با بهینه‌سازی حافظه"""
        if df.empty or 'price' not in df.columns:
            return df
        
        try:
            # فقط اندیکاتورهای ضروری را محاسبه کن
            result_df = df.copy()
            price = result_df['price']
            
            # 1. اندیکاتورهای پایه (کم مصرف)
            if self.indicator_config.get('enable_basic_indicators', True):
                # میانگین‌های متحرک
                sma_periods = self.indicator_config.get('sma_periods', [20, 50])
                for period in sma_periods:
                    result_df[f'sma_{period}'] = ta.sma(price, length=period)
                
                # RSI
                rsi_period = self.indicator_config.get('rsi_period', 14)
                result_df[f'rsi_{rsi_period}'] = ta.rsi(price, length=rsi_period)
            
            # 2. اندیکاتورهای پیشرفته (فقط اگر فعال باشند)
            if self.indicator_config.get('enable_advanced_indicators', False):
                high = result_df.get('high', price)
                low = result_df.get('low', price)
                
                # MACD
                macd = ta.macd(price, fast=12, slow=26, signal=9)
                if macd is not None:
                    result_df['macd'] = macd['MACD_12_26_9']
                    result_df['macd_signal'] = macd['MACDs_12_26_9']
                
                # Bollinger Bands
                bb = ta.bbands(price, length=20, std=2)
                if bb is not None:
                    result_df['bb_upper'] = bb['BBU_20_2.0']
                    result_df['bb_lower'] = bb['BBL_20_2.0']
            
            # 3. سیگنال‌های ترکیبی بهینه‌شده
            result_df = self._calculate_essential_signals(result_df)
            
            # 4. بهینه‌سازی نهایی حافظه
            result_df = self.optimize_dataframe_memory(result_df)
            
            # پاک‌سازی حافظه
            self._cleanup_memory()
            
            print(f"✅ اندیکاتورها محاسبه شد - حافظه: {self._get_memory_usage()}")
            return result_df
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورها: {e}")
            return self._calculate_basic_indicators_fallback(df)
    
    def _calculate_essential_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه سیگنال‌های ضروری با حداقل حافظه"""
        try:
            # سیگنال RSI
            if 'rsi_14' in df.columns:
                df['rsi_signal'] = np.select([
                    df['rsi_14'] > 70,
                    df['rsi_14'] < 30
                ], ['اشباع خرید', 'اشباع فروش'], default='عادی')
            
            # سیگنال میانگین متحرک
            if all(col in df.columns for col in ['sma_20', 'sma_50']):
                df['trend_direction'] = np.where(
                    df['sma_20'] > df['sma_50'], 'صعودی', 'نزولی'
                )
            
            return df
            
        except Exception as e:
            print(f"⚠️ خطا در سیگنال‌های ضروری: {e}")
            return df
    
    def _process_large_data_in_chunks(self, chart_data: List[Dict]) -> pd.DataFrame:
        """پردازش داده‌های بزرگ به صورت chunk"""
        try:
            chunk_size = self.mem_config.get('chunk_size', 1000)
            chunks = [chart_data[i:i + chunk_size] for i in range(0, len(chart_data), chunk_size)]
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"🔨 پردازش chunk {i+1}/{len(chunks)}")
                
                chunk_df = pd.DataFrame(chunk)
                chunk_df = self.optimize_dataframe_memory(chunk_df)
                
                # تبدیل timestamp
                if 'timestamp' in chunk_df.columns:
                    chunk_df['datetime'] = pd.to_datetime(chunk_df['timestamp'], unit='s')
                
                # پردازش عددی
                numeric_columns = ['price', 'volume', 'high', 'low', 'open']
                for col in numeric_columns:
                    if col in chunk_df.columns:
                        chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                
                processed_chunks.append(chunk_df)
                
                # پاک‌سازی حافظه بعد از هر chunk
                if i % 5 == 0:
                    self._cleanup_memory()
            
            # ترکیب chunkها
            final_df = pd.concat(processed_chunks, ignore_index=True)
            final_df = final_df.sort_values('datetime').reset_index(drop=True)
            final_df = final_df.dropna(subset=['price'])
            
            return final_df
            
        except Exception as e:
            print(f"❌ خطا در پردازش chunk-based: {e}")
            return pd.DataFrame()
    
    def _calculate_basic_indicators_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبات پایه fallback با حداقل حافظه"""
        try:
            result_df = df.copy()
            price = result_df['price']
            
            # فقط ضروری‌ترین اندیکاتورها
            result_df['sma_20'] = price.rolling(window=20).mean()
            result_df['sma_50'] = price.rolling(window=50).mean()
            result_df['rsi_14'] = self.calculate_rsi_memory_efficient(price, 14)
            
            return self.optimize_dataframe_memory(result_df)
            
        except Exception as e:
            print(f"❌ خطا در محاسبات پایه: {e}")
            return df
    
    def calculate_rsi_memory_efficient(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه RSI با مصرف حافظه بهینه"""
        try:
            delta = prices.diff()
            
            # استفاده از rolling با min_periods برای صرفه‌جویی در حافظه
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # مقدار پیش‌فرض برای داده‌های ناکافی
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _cleanup_memory(self):
        """پاک‌سازی حافظه"""
        self.processing_count += 1
        
        # هر چند وقت یکبار GC را فراخوانی کن
        if self.processing_count % self.mem_config.get('cleanup_interval', 100) == 0:
            gc.collect()
            print("🧹 حافظه پاک‌سازی شد")
    
    def _get_memory_usage(self) -> str:
        """دریافت میزان استفاده از حافظه"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        except:
            return "نامشخص"
    
    def clear_cache(self):
        """پاک‌سازی کامل کش"""
        self.processed_data.clear()
        self.technical_indicators.clear()
        gc.collect()
        print("✅ کش حافظه پاک‌سازی شد")

# تست بهینه‌سازی
if __name__ == "__main__":
    processor = OptimizedDataProcessor()
    
    # داده‌های نمونه بزرگ
    large_sample = [
        {"timestamp": 1703000000 + i*3600, "price": 45000 + i*100, "volume": 1000000 + i*10000}
        for i in range(2000)  # 2000 نقطه داده
    ]
    
    processed = processor.process_chart_data_optimized(large_sample)
    print(f"داده‌های پردازش شده: {len(processed)} رکورد")
    
    if not processed.empty:
        with_indicators = processor.calculate_technical_indicators_optimized(processed)
        print(f"اندیکاتورهای محاسبه شده: {list(with_indicators.columns)}")
