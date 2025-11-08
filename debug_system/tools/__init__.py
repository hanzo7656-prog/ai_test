"""
Debug System Tools
Development and testing utilities
"""

import logging
from ..core import debug_manager
from ..storage import history_manager
from .dev_tools import DevTools
from .testing_tools import TestingTools
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های ابزار با Dependency Injection
dev_tools = DevTools(debug_manager, None)  # endpoint_monitor بعداً تنظیم می‌شود
testing_tools = TestingTools(debug_manager, None)  # endpoint_monitor بعداً تنظیم می‌شود
report_generator = ReportGenerator(debug_manager, history_manager)

def initialize_tools_system(endpoint_monitor_instance=None):
    """راه‌اندازی و ارتباط ابزارهای توسعه"""
    try:
        # تنظیم endpoint monitor برای ابزارها
        if endpoint_monitor_instance:
            dev_tools.endpoint_monitor = endpoint_monitor_instance
            testing_tools.endpoint_monitor = endpoint_monitor_instance
            logger.info("✅ Endpoint monitor connected to development tools")
        
        logger.info("✅ Debug tools system initialized with dependency injection")
        logger.info(f"   - Dev Tools: {type(dev_tools).__name__}")
        logger.info(f"   - Testing Tools: {type(testing_tools).__name__}")
        logger.info(f"   - Report Generator: {type(report_generator).__name__}")
        
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools,
            "report_generator": report_generator
        }
    except Exception as e:
        logger.error(f"❌ Tools initialization failed: {e}")
        raise

# راه‌اندازی اولیه (بدون endpoint_monitor)
tools_system = initialize_tools_system()

__all__ = [
    "DevTools", "dev_tools",
    "TestingTools", "testing_tools",
    "ReportGenerator", "report_generator",
    "initialize_tools_system", "tools_system"
]
