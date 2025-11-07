# تنظیمات مرکزی Trading AI
from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class AIConfig:
    """کلاس مدیریت تنظیمات هوش مصنوعی"""
    
    # تنظیمات شبکه عصبی
    NEURAL_NETWORK = {
        'input_size': 20,
        'hidden_size': 100,  # 100 نورون
        'output_size': 5,    # 5 سیگنال خروجی
        'sparsity': 0.8,     # 80% اسپارس
        'learning_rate': 0.01,
        'epochs': 100,
        'batch_size': 32
    }
    
    # تنظیمات تحلیل تکنیکال
    TECHNICAL_ANALYSIS = {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2
    }
    
    # تنظیمات سیگنال‌ها
    SIGNALS = {
        'STRONG_BUY': {'min_confidence': 0.8, 'color': '#00d9a6'},
        'BUY': {'min_confidence': 0.6, 'color': '#00b894'},
        'HOLD': {'min_confidence': 0.4, 'color': '#ff9f43'},
        'SELL': {'min_confidence': 0.6, 'color': '#ff6b6b'},
        'STRONG_SELL': {'min_confidence': 0.8, 'color': '#ff4757'}
    }
    
    # تنظیمات مسیرها
    PATHS = {
        'models_dir': 'trading_ai/models',
        'data_dir': 'trading_ai/data',
        'logs_dir': 'trading_ai/logs',
        'cache_dir': 'trading_ai/cache'
    }
    
    # تنظیمات عملکرد
    PERFORMANCE = {
        'max_symbols_per_batch': 50,
        'request_timeout': 15,
        'cache_ttl': 300,  # 5 دقیقه
        'retry_attempts': 3
    }
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.custom_config = {}
        
        if config_file:
            self.load_config(config_file)
        
        # ایجاد پوشه‌ها
        self._create_directories()
    
    def _create_directories(self):
        """ایجاد پوشه‌های لازم"""
        for path_key, path_value in self.PATHS.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ پوشه‌های Trading AI ایجاد شدند")
    
    def load_config(self, config_file: str):
        """بارگذاری تنظیمات از فایل"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.custom_config = json.load(f)
            
            logger.info(f"✅ تنظیمات از {config_file} بارگذاری شد")
            
        except Exception as e:
            logger.warning(f"⚠️ خطا در بارگذاری تنظیمات: {e}")
    
    def save_config(self, config_file: str = None):
        """ذخیره تنظیمات در فایل"""
        try:
            save_path = config_file or self.config_file
            if not save_path:
                logger.warning("⚠️ مسیر فایل تنظیمات مشخص نشده")
                return
            
            config_data = {
                'neural_network': self.NEURAL_NETWORK,
                'technical_analysis': self.TECHNICAL_ANALYSIS,
                'signals': self.SIGNALS,
                'performance': self.PERFORMANCE
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 تنظیمات در {save_path} ذخیره شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره تنظیمات: {e}")
    
    def get(self, section: str, key: str = None, default=None):
        """دریافت مقدار تنظیمات"""
        try:
            # اول تنظیمات سفارشی
            if section in self.custom_config:
                section_data = self.custom_config[section]
                if key:
                    return section_data.get(key, getattr(self, section.upper(), {}).get(key, default))
                return section_data
            
            # سپس تنظیمات پیش‌فرض
            section_data = getattr(self, section.upper(), {})
            if key:
                return section_data.get(key, default)
            return section_data
            
        except Exception as e:
            logger.error(f"خطا در دریافت تنظیمات {section}.{key}: {e}")
            return default
    
    def update(self, section: str, key: str, value: Any):
        """بروزرسانی تنظیمات"""
        try:
            if section not in self.custom_config:
                self.custom_config[section] = {}
            
            self.custom_config[section][key] = value
            logger.info(f"⚙️ تنظیمات {section}.{key} بروزرسانی شد")
            
        except Exception as e:
            logger.error(f"خطا در بروزرسانی تنظیمات: {e}")
    
    def get_neural_network_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات شبکه عصبی"""
        return self.get('neural_network')
    
    def get_technical_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات تحلیل تکنیکال"""
        return self.get('technical_analysis')
    
    def get_signal_config(self, signal_type: str = None) -> Dict[str, Any]:
        """دریافت تنظیمات سیگنال"""
        signals = self.get('signals')
        if signal_type:
            return signals.get(signal_type, {})
        return signals
    
    def validate_config(self) -> bool:
        """اعتبارسنجی تنظیمات"""
        try:
            # اعتبارسنجی شبکه عصبی
            nn_config = self.get_neural_network_config()
            assert nn_config['hidden_size'] > 0, "تعداد نورون‌ها باید مثبت باشد"
            assert 0 <= nn_config['sparsity'] <= 1, "میزان اسپارسیتی باید بین 0 و 1 باشد"
            
            # اعتبارسنجی تحلیل تکنیکال
            ta_config = self.get_technical_config()
            assert ta_config['rsi_period'] > 0, "دوره RSI باید مثبت باشد"
            assert 0 < ta_config['rsi_oversold'] < ta_config['rsi_overbought'] < 100, "مقادیر RSI نامعتبر"
            
            logger.info("✅ تنظیمات معتبر هستند")
            return True
            
        except AssertionError as e:
            logger.error(f"❌ تنظیمات نامعتبر: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ خطا در اعتبارسنجی تنظیمات: {e}")
            return False

# نمونه جهانی
ai_config = AIConfig()
