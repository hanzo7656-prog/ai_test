# database_manager.py - مدیریت پایگاه داده ساده برای تست با داده‌های خام

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json

logger = logging.getLogger(__name__)

class TradingDatabase:
    """پایگاه داده ساده برای داده‌های تاریخی با داده‌های خام"""
    
    def __init__(self):
        self.data_dir = "./trading_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.raw_data_cache = {}
        self.cache_expiry = 300  # 5 دقیقه
        logger.info("🚀 Trading Database Initialized - Raw Data Mode")

    def get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """دریافت داده‌های تاریخی - نسخه ساده برای تست با داده‌های خام"""
        try:
            logger.info(f"📊 دریافت داده‌های تاریخی {symbol} برای {days} روز")

            # اگر فایل ذخیره شده وجود دارد، از آن استفاده کن
            file_path = os.path.join(self.data_dir, f"{symbol}_historical.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                if len(df) >= days:
                    logger.info(f"✅ داده‌های کش شده {symbol} بارگذاری شد: {len(df)} رکورد")
                    return df.tail(days)

            # در غیر این صورت داده نمونه تولید کن
            return self._generate_sample_data(symbol, days)

        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های {symbol}: {e}")
            return self._generate_sample_data(symbol, days)

    def _generate_sample_data(self, symbol: str, days: int) -> pd.DataFrame:
        """تولید داده نمونه واقعی‌تر با داده‌های خام"""
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

        # تولید داده‌های خام OHLCV
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
        """ذخیره داده‌های بازار خام"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_market.json")
            
            # اضافه کردن metadata
            enhanced_data = {
                'symbol': symbol,
                'saved_at': datetime.now().isoformat(),
                'data_source': 'raw_market_data',
                'raw_data': data
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ داده‌های بازار {symbol} ذخیره شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره داده‌های {symbol}: {e}")

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """دریافت داده‌های بازار ذخیره شده خام"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_market.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # بررسی تازگی داده‌ها (کمتر از 1 ساعت)
                saved_time = datetime.fromisoformat(data.get('saved_at', ''))
                if datetime.now() - saved_time < timedelta(hours=1):
                    logger.info(f"✅ داده‌های بازار {symbol} بازیابی شد")
                    return data
                else:
                    logger.warning(f"⚠️ داده‌های بازار {symbol} منقضی شده")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ خطا در خواندن داده‌های {symbol}: {e}")
            
        return None

    def save_technical_analysis(self, symbol: str, analysis_data: Dict):
        """ذخیره نتایج تحلیل تکنیکال"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_analysis.json")
            
            enhanced_analysis = {
                'symbol': symbol,
                'analyzed_at': datetime.now().isoformat(),
                'analysis_type': 'technical',
                'raw_indicators': analysis_data,
                'data_quality': self._assess_analysis_quality(analysis_data)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ تحلیل تکنیکال {symbol} ذخیره شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره تحلیل {symbol}: {e}")

    def get_technical_analysis(self, symbol: str) -> Optional[Dict]:
        """دریافت تحلیل تکنیکال ذخیره شده"""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_analysis.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"❌ خطا در خواندن تحلیل {symbol}: {e}")
            
        return None

    def _assess_analysis_quality(self, analysis_data: Dict) -> Dict[str, Any]:
        """ارزیابی کیفیت داده‌های تحلیل"""
        quality_metrics = {
            'completeness': 0.0,
            'freshness': 1.0,  # داده‌های جدید
            'consistency': 0.8,
            'overall_score': 0.0
        }
        
        try:
            # بررسی کامل بودن داده‌ها
            required_fields = ['trend', 'levels', 'indicators']
            present_fields = [field for field in required_fields if field in analysis_data]
            quality_metrics['completeness'] = len(present_fields) / len(required_fields)
            
            # محاسبه نمره کلی
            quality_metrics['overall_score'] = round(
                (quality_metrics['completeness'] * 0.4 +
                 quality_metrics['freshness'] * 0.3 +
                 quality_metrics['consistency'] * 0.3), 3
            )
            
            quality_metrics['quality_level'] = (
                'high' if quality_metrics['overall_score'] > 0.8 else
                'medium' if quality_metrics['overall_score'] > 0.5 else 'low'
            )
            
        except Exception as e:
            logger.error(f"❌ خطا در ارزیابی کیفیت: {e}")
            
        return quality_metrics

    def get_database_stats(self) -> Dict[str, Any]:
        """دریافت آمار پایگاه داده"""
        try:
            stats = {
                'total_symbols': 0,
                'total_files': 0,
                'total_size_mb': 0,
                'data_types': {},
                'last_updated': datetime.now().isoformat()
            }
            
            if os.path.exists(self.data_dir):
                for file in os.listdir(self.data_dir):
                    if file.endswith(('.csv', '.json')):
                        stats['total_files'] += 1
                        file_path = os.path.join(self.data_dir, file)
                        stats['total_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
                        
                        # طبقه‌بندی فایل‌ها
                        if 'historical' in file:
                            stats['data_types']['historical'] = stats['data_types'].get('historical', 0) + 1
                        elif 'market' in file:
                            stats['data_types']['market'] = stats['data_types'].get('market', 0) + 1
                        elif 'analysis' in file:
                            stats['data_types']['analysis'] = stats['data_types'].get('analysis', 0) + 1
                
                # تعداد نمادهای منحصر به فرد
                symbols = set()
                for file in os.listdir(self.data_dir):
                    if '_' in file:
                        symbol = file.split('_')[0]
                        symbols.add(symbol)
                stats['total_symbols'] = len(symbols)
                
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            return stats
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت آمار پایگاه داده: {e}")
            return {'error': str(e)}

    def clear_old_data(self, days_old: int = 30):
        """پاکسازی داده‌های قدیمی"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            deleted_files = 0
            
            for file in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, file)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        deleted_files += 1
                        logger.info(f"🧹 فایل قدیمی پاکسازی شد: {file}")
            
            logger.info(f"✅ {deleted_files} فایل قدیمی پاکسازی شد")
            return deleted_files
            
        except Exception as e:
            logger.error(f"❌ خطا در پاکسازی داده‌های قدیمی: {e}")
            return 0

    def backup_database(self, backup_dir: str = "./backup"):
        """پشتیبان‌گیری از پایگاه داده"""
        try:
            import shutil
            import datetime
            
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # ایجاد نام فایل پشتیبان با timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"trading_db_backup_{timestamp}")
            
            # کپی کردن دایرکتوری
            if os.path.exists(self.data_dir):
                shutil.copytree(self.data_dir, backup_path)
                logger.info(f"✅ پشتیبان‌گیری انجام شد: {backup_path}")
                return backup_path
            else:
                logger.warning("⚠️ دایرکتوری داده‌ها برای پشتیبان‌گیری وجود ندارد")
                return None
                
        except Exception as e:
            logger.error(f"❌ خطا در پشتیبان‌گیری: {e}")
            return None

# ایجاد نمونه گلوبال
trading_db = TradingDatabase()
