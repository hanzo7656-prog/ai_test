"""
Debug System Tools
Development and testing utilities
"""

import logging

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های خالی - در initialize پر می‌شوند
dev_tools = None
testing_tools = None
report_generator = None

def initialize_tools_system(debug_manager_instance=None, history_manager_instance=None):
    """راه‌اندازی و ارتباط ابزارهای توسعه"""
    try:
        from .dev_tools import DevTools
        from .testing_tools import TestingTools
        from .report_generator import ReportGenerator
        
        global dev_tools, testing_tools, report_generator
        
        # ایجاد نمونه‌ها با dependency injection
        if debug_manager_instance:
            dev_tools = DevTools(debug_manager_instance)
            testing_tools = TestingTools(debug_manager_instance)
        
        if debug_manager_instance and history_manager_instance:
            report_generator = ReportGenerator(debug_manager_instance, history_manager_instance)
        
        logger.info("✅ Debug tools system initialized")
        logger.info(f"   - Dev Tools: {type(dev_tools).__name__ if dev_tools else 'Not available'}")
        logger.info(f"   - Testing Tools: {type(testing_tools).__name__ if testing_tools else 'Not available'}")
        logger.info(f"   - Report Generator: {type(report_generator).__name__ if report_generator else 'Not available'}")
        
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools,
            "report_generator": report_generator
        }
    except Exception as e:
        logger.error(f"❌ Tools initialization failed: {e}")
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools,
            "report_generator": report_generator
        }

# راه‌اندازی اولیه (بدون dependency)
tools_system = initialize_tools_system()

__all__ = [
    "DevTools", "dev_tools",
    "TestingTools", "testing_tools",
    "ReportGenerator", "report_generator",
    "initialize_tools_system", "tools_system"
]
