# ml_core/__init__.py
"""
هسته اصلی سیستم هوش مصنوعی - ML Core
ماژول مدیریت مدل‌ها، مانیتورینگ سلامت، یکپارچه‌سازی داده و ردیابی عملکرد
"""

from .model_manager import ml_model_manager, MLModelManager
from .health_monitor import ml_health_monitor, initialize_health_monitor
from .data_integration import data_integrator, DataIntegration
from .performance_tracker import performance_tracker, initialize_performance_tracker

# وارد کردن تحلیل‌گر تکنیکال اسپارس
try:
    from .technical_analyzer import SparseTechnicalNetwork, SparseConfig, TechnicalAnalysisTrainer
except ImportError:
    # برای زمانی که فایل تحلیل‌گر هنوز اضافه نشده
    pass

__all__ = [
    # کلاس‌های اصلی
    'MLModelManager',
    'DataIntegration',
    
    # نمونه‌های global
    'ml_model_manager',
    'data_integrator',
    
    # توابع مقداردهی
    'initialize_health_monitor',
    'initialize_performance_tracker',
    
    # تحلیل‌گر تکنیکال
    'SparseTechnicalNetwork',
    'SparseConfig', 
    'TechnicalAnalysisTrainer'
]

# مقداردهی اولیه کامپوننت‌ها هنگام ایمپورت
def initialize_ml_core():
    """مقداردهی اولیه کامل هسته ML"""
    try:
        # مقداردهی مانیتور سلامت
        health_monitor = initialize_health_monitor(ml_model_manager)
        
        # مقداردهی performance tracker
        perf_tracker = initialize_performance_tracker(ml_model_manager)
        
        print("✅ ML Core initialized successfully!")
        print(f"   - Active models: {len(ml_model_manager.active_models)}")
        print(f"   - Data integrator: Ready")
        print(f"   - Health monitor: Ready") 
        print(f"   - Performance tracker: Ready")
        
        return {
            'model_manager': ml_model_manager,
            'health_monitor': health_monitor,
            'performance_tracker': perf_tracker,
            'data_integrator': data_integrator
        }
        
    except Exception as e:
        print(f"❌ Error initializing ML Core: {e}")
        raise

# اجرای خودکار مقداردهی اولیه هنگام ایمپورت
try:
    initialize_ml_core()
except Exception as e:
    print(f"⚠️ ML Core auto-initialization skipped: {e}")
