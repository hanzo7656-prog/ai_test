# 📁 config/__init__.py

import os
import yaml
from typing import Dict, Any

def load_config(environment: str = None) -> Dict[str, Any]:
    """لود پیکربندی بر اساس محیط اجرا"""
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    config_files = {
        'development': 'development.yaml',
        'production': 'production.yaml',
        'testing': 'testing.yaml'
    }
    
    config_file = config_files.get(environment, 'development.yaml')
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # بارگذاری phase5 config
        from .phase5_config import PHASE5_CONFIG
        config['phase5'] = PHASE5_CONFIG
        
        return config
        
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default configuration.")
        from .phase5_config import PHASE5_CONFIG
        return {'phase5': PHASE5_CONFIG}
    
    except Exception as e:
        print(f"Error loading config: {e}")
        from .phase5_config import PHASE5_CONFIG
        return {'phase5': PHASE5_CONFIG}

# پیکربندی پیش‌فرض
current_config = load_config()
