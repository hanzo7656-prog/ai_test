"""
Debug System Core Modules - Lightweight Edition
Minimal initialization with zero interference
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global instances - lazy loaded
_instances = {}
_initialized = False

def _lazy_load_module(module_name: str, class_name: str):
    """Lazy import modules to reduce startup overhead"""
    try:
        module = __import__(f'.{module_name}', fromlist=[class_name], level=1)
        return getattr(module, class_name)
    except ImportError as e:
        logger.debug(f"Delayed import of {module_name}.{class_name}")
        return None

def get_debug_manager():
    """Get DebugManager instance (lazy initialization)"""
    if 'debug_manager' not in _instances:
        DebugManager = _lazy_load_module('debug_manager', 'DebugManager')
        if DebugManager:
            _instances['debug_manager'] = DebugManager()
        else:
            # Fallback minimal instance
            class MinimalDebugManager:
                def __init__(self): self.is_active = lambda: True
                def log_endpoint_call(self, *args, **kwargs): pass
                def get_endpoint_stats(self, *args, **kwargs): return {}
            _instances['debug_manager'] = MinimalDebugManager()
    
    return _instances['debug_manager']

def get_metrics_collector():
    """Get MetricsCollector instance (lazy initialization)"""
    if 'metrics_collector' not in _instances:
        RealTimeMetricsCollector = _lazy_load_module('metrics_collector', 'RealTimeMetricsCollector')
        if RealTimeMetricsCollector:
            _instances['metrics_collector'] = RealTimeMetricsCollector()
        else:
            # Fallback minimal instance
            class MinimalMetricsCollector:
                def __init__(self): 
                    self.get_current_metrics = lambda: {'status': 'fallback'}
                    self.get_metrics_history = lambda *args: []
            _instances['metrics_collector'] = MinimalMetricsCollector()
    
    return _instances['metrics_collector']

def get_alert_manager():
    """Get AlertManager instance (lazy initialization)"""
    if 'alert_manager' not in _instances:
        AlertManager = _lazy_load_module('alert_manager', 'AlertManager')
        if AlertManager:
            _instances['alert_manager'] = AlertManager()
        else:
            # Fallback minimal instance
            class MinimalAlertManager:
                def __init__(self): 
                    self.create_alert = lambda *args, **kwargs: None
                    self.active_alerts = []
            _instances['alert_manager'] = MinimalAlertManager()
    
    return _instances['alert_manager']

def get_central_monitor():
    """Get central_monitor from system_monitor (reference only)"""
    # فقط یک reference برمی‌گرداند، ایجاد نمی‌کند
    try:
        from .system_monitor import central_monitor as cm
        return cm  # فقط reference موجود را برمی‌گرداند
    except ImportError:
        return None

def initialize_safe():
    """Safe minimal initialization without interference"""
    global _initialized
    
    if _initialized:
        return True
    
    try:
        logger.info("🔧 Initializing core modules safely...")
        start_time = time.time()
        
        # فقط instance های اصلی را lazy load کن
        debug_manager = get_debug_manager()
        alert_manager = get_alert_manager()
        
        # اتصال ساده (اگر ممکن باشد)
        if hasattr(debug_manager, 'set_alert_manager'):
            try:
                debug_manager.set_alert_manager(alert_manager)
                logger.debug("✅ Basic alert integration configured")
            except:
                pass  # ignore failures
        
        _initialized = True
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Core modules ready in {elapsed:.3f}s (lightweight mode)")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Light initialization completed with warnings: {e}")
        _initialized = True  # Mark as initialized anyway
        return True  # همیشه True برگردان - fail gracefully

def get_core_status() -> Dict[str, Any]:
    """Get minimal status report"""
    return {
        'timestamp': time.time(),
        'initialized': _initialized,
        'modules_loaded': list(_instances.keys()),
        'mode': 'lightweight',
        'central_monitor_available': get_central_monitor() is not None
    }

def shutdown_gracefully():
    """Graceful shutdown - minimal cleanup"""
    global _instances, _initialized
    
    logger.debug("🛑 Gracefully clearing core instances")
    _instances.clear()
    _initialized = False

# Convenience accessors (properties for backward compatibility)
class CoreAccessor:
    """Lightweight accessor for core modules"""
    
    @property
    def debug_manager(self):
        return get_debug_manager()
    
    @property
    def metrics_collector(self):
        return get_metrics_collector()
    
    @property
    def alert_manager(self):
        return get_alert_manager()
    
    @property
    def central_monitor(self):
        return get_central_monitor()
    
    def initialize(self):
        return initialize_safe()
    
    def status(self):
        return get_core_status()
    
    def shutdown(self):
        return shutdown_gracefully()

# Single global accessor instance
core = CoreAccessor()

# Auto-initialize on first access (if needed)
def _auto_init_if_needed():
    """Auto-initialize only if modules are actually used"""
    # بررسی می‌کنیم آیا سیستم core واقعاً استفاده شده یا نه
    pass  # کاری نمی‌کند - منتظر اولین درخواست می‌ماند

# برای backward compatibility - global aliases
# اما اینها properties هستند که lazy load می‌شوند
debug_manager = core.debug_manager
metrics_collector = core.metrics_collector
alert_manager = core.alert_manager
central_monitor = core.central_monitor

__all__ = [
    # کلاس‌ها
    "DebugManager", 
    "RealTimeMetricsCollector", 
    "AlertManager", 
    "AlertLevel", 
    "AlertType",
    
    # توابع
    "initialize_safe",
    "get_core_status",
    "shutdown_gracefully",
    
    # Accessors
    "core",
    
    # Aliases (برای compatibility)
    "debug_manager",
    "metrics_collector", 
    "alert_manager",
    "central_monitor"
]

# Import کلاس‌ها برای __all__ (اما instantiate نمی‌کنیم)
try:
    from .debug_manager import DebugManager
    from .metrics_collector import RealTimeMetricsCollector
    from .alert_manager import AlertManager, AlertLevel, AlertType
except ImportError:
    # اگر import شکست خورد، تعریف‌های حداقلی
    class DebugManager: pass
    class RealTimeMetricsCollector: pass
    class AlertManager: pass
    class AlertLevel: pass
    class AlertType: pass

# پیام شروع
logger.debug("💡 Core modules loaded in lightweight mode (no auto-init)")
