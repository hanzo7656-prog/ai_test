import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """بارگذاری تمام فایل‌های کانفیگ"""
        try:
            # ایجاد پوشه config اگر وجود ندارد
            self.config_dir.mkdir(exist_ok=True)
            
            # بارگذاری فایل‌های YAML
            for yaml_file in self.config_dir.glob("*.yaml"):
                self._load_yaml_config(yaml_file)
            
            # بارگذاری فایل‌های YML
            for yml_file in self.config_dir.glob("*.yml"):
                self._load_yaml_config(yml_file)
            
            # بارگذاری فایل‌های JSON
            for json_file in self.config_dir.glob("*.json"):
                self._load_json_config(json_file)
            
            logger.info(f"✅ Loaded {len(self.configs)} configuration files")
            
        except Exception as e:
            logger.error(f"❌ Error loading configurations: {e}")
    
    def _load_yaml_config(self, file_path: Path):
        """بارگذاری فایل YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_name = file_path.stem
                self.configs[config_name] = yaml.safe_load(f)
                logger.debug(f"📁 Loaded YAML config: {config_name}")
        except Exception as e:
            logger.error(f"❌ Error loading YAML config {file_path}: {e}")
    
    def _load_json_config(self, file_path: Path):
        """بارگذاری فایل JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_name = file_path.stem
                self.configs[config_name] = json.load(f)
                logger.debug(f"📁 Loaded JSON config: {config_name}")
        except Exception as e:
            logger.error(f"❌ Error loading JSON config {file_path}: {e}")
    
    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """دریافت مقدار کانفیگ"""
        try:
            config = self.configs.get(config_name, {})
            
            if key is None:
                return config
            
            # پشتیبانی از keys تودرتو با dot notation
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, {})
                else:
                    return default
            
            return value if value != {} else default
            
        except Exception as e:
            logger.error(f"❌ Error getting config {config_name}.{key}: {e}")
            return default
    
    def set(self, config_name: str, key: str, value: Any):
        """تنظیم مقدار کانفیگ"""
        try:
            if config_name not in self.configs:
                self.configs[config_name] = {}
            
            # پشتیبانی از keys تودرتو با dot notation
            keys = key.split('.')
            config = self.configs[config_name]
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            logger.debug(f"📝 Set config: {config_name}.{key} = {value}")
            
        except Exception as e:
            logger.error(f"❌ Error setting config {config_name}.{key}: {e}")
    
    def save_config(self, config_name: str, format: str = "yaml"):
        """ذخیره کانفیگ در فایل"""
        try:
            if config_name not in self.configs:
                logger.warning(f"Config {config_name} not found")
                return False
            
            if format == "yaml":
                file_path = self.config_dir / f"{config_name}.yaml"
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.configs[config_name], f, default_flow_style=False, allow_unicode=True)
            
            elif format == "json":
                file_path = self.config_dir / f"{config_name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.configs[config_name], f, indent=2, ensure_ascii=False)
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"💾 Saved config: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving config {config_name}: {e}")
            return False
    
    def reload(self):
        """بارگذاری مجدد تمام کانفیگ‌ها"""
        self.configs.clear()
        self._load_all_configs()
        logger.info("🔄 Configurations reloaded")
    
    def list_configs(self) -> Dict[str, Any]:
        """لیست تمام کانفیگ‌های موجود"""
        return {
            'loaded_configs': list(self.configs.keys()),
            'config_files': [f.name for f in self.config_dir.glob('*')],
            'config_dir': str(self.config_dir)
        }

# ایجاد نمونه گلوبال
config_loader = ConfigLoader()
