"""
Debug System Tools Package
ابزارهای توسعه، تست و گزارش‌گیری
"""

import logging

logger = logging.getLogger(__name__)

# نمونه‌های خالی - در initialize پر می‌شوند
dev_tools = None
testing_tools = None
report_generator = None

def initialize_tools_system(debug_manager_instance=None, history_manager_instance=None):
    """راه‌اندازی و ارتباط ابزارهای توسعه"""
    try:
        # Lazy import برای جلوگیری از circular dependency
        from .dev_tools import DevTools
        from .testing_tools import TestingTools
        from .report_generator import ReportGenerator
        
        global dev_tools, testing_tools, report_generator
        
        # ایجاد نمونه‌ها با dependency injection
        if debug_manager_instance:
            dev_tools = DevTools(debug_manager_instance)
            testing_tools = TestingTools(debug_manager_instance)
            logger.info("✅ DevTools and TestingTools initialized")
        
        if debug_manager_instance and history_manager_instance:
            report_generator = ReportGenerator(debug_manager_instance, history_manager_instance)
            logger.info("✅ ReportGenerator initialized")
        
        logger.info("🎯 Debug tools system fully initialized")
        
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools, 
            "report_generator": report_generator,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Tools initialization failed: {e}")
        # ایجاد stub برای جلوگیری از خطا
        class StubTools:
            def __getattr__(self, name):
                return lambda *args, **kwargs: {"error": "Tools not initialized"}
        
        if debug_manager_instance and not dev_tools:
            dev_tools = StubTools()
            testing_tools = StubTools()
        
        if debug_manager_instance and history_manager_instance and not report_generator:
            report_generator = StubTools()
        
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools,
            "report_generator": report_generator,
            "status": "partial",
            "error": str(e)
        }

# ایمپورت کلاس‌ها برای export
try:
    from .dev_tools import DevTools
    from .testing_tools import TestingTools
    from .report_generator import ReportGenerator
except ImportError as e:
    logger.warning(f"⚠️ Could not import tools classes: {e}")
    
    # ایجاد stub classes
    class DevTools:
        def __init__(self, debug_manager=None):
            self.debug_manager = debug_manager
    
    class TestingTools:
        def __init__(self, debug_manager=None):
            self.debug_manager = debug_manager
    
    class ReportGenerator:
        def __init__(self, debug_manager=None, history_manager=None):
            self.debug_manager = debug_manager
            self.history_manager = history_manager

# Fallback برای tools_system
tools_system = {
    "dev_tools": dev_tools,
    "testing_tools": testing_tools, 
    "report_generator": report_generator,
    "initialize": initialize_tools_system
}

__all__ = [
    "DevTools", "dev_tools",
    "TestingTools", "testing_tools", 
    "ReportGenerator", "report_generator",
    "initialize_tools_system", "tools_system"
]
