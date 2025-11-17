import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AIConfig:
    """تنظیمات مرکزی هوش مصنوعی"""
    
    def __init__(self):
        self.config = self._load_config()
        logger.info("🚀 تنظیمات هوش مصنوعی بارگذاری شد")
    
    def _load_config(self) -> Dict[str, Any]:
        """بارگذاری تنظیمات از environment variables و مقادیر پیش‌فرض"""
        
        base_config = {
            # تنظیمات اصلی سیستم
            'system': {
                'name': 'VortexAI Brain',
                'version': '1.0.0',
                'description': 'هوش مصنوعی اسپارس خودآموز برای تحلیل بازار کریپتو',
                'base_url': os.getenv('AI_BASE_URL', 'https://ai-test-3gix.onrender.com'),
                'environment': os.getenv('AI_ENVIRONMENT', 'development')
            },
            
            # تنظیمات شبکه عصبی
            'neural_network': {
                'num_neurons': 1000,
                'sparsity': 0.1,
                'learning_rate': 0.01,
                'max_complexity': 50,
                'activation_threshold': 0.1
            },
            
            # تنظیمات پردازش متن
            'text_processing': {
                'max_vocab_size': 2000,
                'language': 'multi',
                'supported_languages': ['fa', 'en'],
                'stop_words_auto_update': True
            },
            
            # تنظیمات یادگیری
            'learning': {
                'learning_rate': 0.01,
                'min_learning_threshold': 0.1,
                'mastery_threshold': 0.7,
                'forgetting_factor': 0.99,
                'max_history_age_days': 30,
                'max_learning_memory_mb': 50
            },
            
            # تنظیمات حافظه
            'memory': {
                'sensory_ttl_hours': 24,
                'working_ttl_days': 30,
                'access_threshold': 3,
                'importance_threshold': 0.7,
                'compression_threshold': 0.8,
                'min_importance_to_keep': 0.3
            },
            
            # تنظیمات API
            'api': {
                'timeout_seconds': 30.0,
                'max_retries': 3,
                'retry_delay': 1.0,
                'health_check_interval': 300
            },
            
            # تنظیمات پاسخ‌دهی
            'response': {
                'default_language': 'fa',
                'use_emojis': True,
                'max_response_length': 1000,
                'include_timestamps': True
            },
            
            # تنظیمات مانیتورینگ
            'monitoring': {
                'enable_health_checks': True,
                'log_level': os.getenv('AI_LOG_LEVEL', 'INFO'),
                'performance_metrics': True,
                'memory_monitoring': True
            },
            
            # تنظیمات ذخیره‌سازی
            'storage': {
                'model_save_path': './ai_brain_storage/model_state.json',
                'backup_enabled': True,
                'backup_interval_hours': 24,
                'auto_save_interval': 3600  # 1 hour
            }
        }
        
        # override با environment variables
        self._apply_env_overrides(base_config)
        
        return base_config
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """اعمال override از environment variables"""
        
        env_mappings = {
            'AI_NUM_NEURONS': ('neural_network', 'num_neurons', int),
            'AI_LEARNING_RATE': ('neural_network', 'learning_rate', float),
            'AI_MAX_COMPLEXITY': ('neural_network', 'max_complexity', int),
            'AI_MEMORY_SENSORY_TTL': ('memory', 'sensory_ttl_hours', int),
            'AI_MEMORY_WORKING_TTL': ('memory', 'working_ttl_days', int),
            'AI_API_TIMEOUT': ('api', 'timeout_seconds', float),
            'AI_DEFAULT_LANGUAGE': ('response', 'default_language', str),
            'AI_LOG_LEVEL': ('monitoring', 'log_level', str)
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    config[section][key] = converter(env_value)
                    logger.info(f"⚙️ تنظیمات از env: {section}.{key} = {env_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ خطا در تبدیل env {env_var}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """دریافت مقدار از مسیر key (مثلاً 'neural_network.num_neurons')"""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """تنظیم مقدار در مسیر key"""
        try:
            keys = key_path.split('.')
            config_ptr = self.config
            for key in keys[:-1]:
                if key not in config_ptr:
                    config_ptr[key] = {}
                config_ptr = config_ptr[key]
            config_ptr[keys[-1]] = value
            logger.debug(f"⚙️ تنظیم شد: {key_path} = {value}")
        except (KeyError, TypeError) as e:
            logger.error(f"❌ خطا در تنظیم {key_path}: {e}")
    
    def get_neural_network_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات شبکه عصبی"""
        return self.config['neural_network']
    
    def get_memory_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات حافظه"""
        return self.config['memory']
    
    def get_learning_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات یادگیری"""
        return self.config['learning']
    
    def get_api_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات API"""
        return self.config['api']
    
    def get_response_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات پاسخ‌دهی"""
        return self.config['response']
    
    def update_from_feedback(self, feedback_data: Dict[str, Any]):
        """به‌روزرسانی تنظیمات بر اساس فیدبک کاربر"""
        
        # تنظیمات مبتنی بر عملکرد
        if 'success_rate' in feedback_data:
            success_rate = feedback_data['success_rate']
            # تنظیم learning_rate بر اساس success_rate
            new_learning_rate = max(0.001, min(0.1, success_rate * 0.1))
            self.set('learning.learning_rate', new_learning_rate)
        
        if 'avg_response_time' in feedback_data:
            response_time = feedback_data['avg_response_time']
            # تنظیم timeout بر اساس زمان پاسخ
            if response_time > 10:
                new_timeout = min(60.0, response_time * 2)
                self.set('api.timeout_seconds', new_timeout)
        
        logger.info("🔄 تنظیمات بر اساس فیدبک به‌روزرسانی شد")
    
    def validate_config(self) -> bool:
        """اعتبارسنجی تنظیمات"""
        try:
            # اعتبارسنجی مقادیر عددی
            assert 100 <= self.get('neural_network.num_neurons') <= 10000
            assert 0 < self.get('neural_network.learning_rate') < 1
            assert 0 < self.get('memory.sensory_ttl_hours') <= 720  # حداکثر 30 روز
            assert 0 < self.get('api.timeout_seconds') <= 300
            
            # اعتبارسنجی مقادیر enum-like
            assert self.get('response.default_language') in ['fa', 'en', 'multi']
            assert self.get('monitoring.log_level') in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            
            logger.info("✅ تنظیمات اعتبارسنجی شدند")
            return True
            
        except (AssertionError, KeyError) as e:
            logger.error(f"❌ خطا در اعتبارسنجی تنظیمات: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """خلاصه تنظیمات"""
        return {
            'system': {
                'name': self.get('system.name'),
                'version': self.get('system.version'),
                'environment': self.get('system.environment')
            },
            'neural_network': {
                'neurons': self.get('neural_network.num_neurons'),
                'sparsity': self.get('neural_network.sparsity'),
                'max_complexity': self.get('neural_network.max_complexity')
            },
            'memory': {
                'sensory_ttl_hours': self.get('memory.sensory_ttl_hours'),
                'working_ttl_days': self.get('memory.working_ttl_days'),
                'compression_threshold': self.get('memory.compression_threshold')
            },
            'api': {
                'timeout': self.get('api.timeout_seconds'),
                'max_retries': self.get('api.max_retries')
            }
        }
    
    def save_to_file(self, filepath: str):
        """ذخیره تنظیمات در فایل"""
        import json
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 تنظیمات در {filepath} ذخیره شد")
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره تنظیمات: {e}")
    
    def load_from_file(self, filepath: str):
        """بارگذاری تنظیمات از فایل"""
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            self.config.update(loaded_config)
            logger.info(f"📂 تنظیمات از {filepath} بارگذاری شد")
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری تنظیمات: {e}")
