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

# ایجاد نمونه‌های ابزار با Dependency Injection صحیح
dev_tools = DevTools(debug_manager)  # ✅ فقط 1 پارامتر - طبق تعریف اصلی
testing_tools = TestingTools(debug_manager)  # ✅ فقط 1 پارامتر - طبق تعریف اصلی
report_generator = ReportGenerator(debug_manager, history_manager)

def initialize_tools_system():
    """راه‌اندازی و ارتباط ابزارهای توسعه"""
    try:
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
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools,
            "report_generator": report_generator
        }

# راه‌اندازی اولیه
tools_system = initialize_tools_system()

__all__ = [
    "DevTools", "dev_tools",
    "TestingTools", "testing_tools",
    "ReportGenerator", "report_generator",
    "initialize_tools_system", "tools_system"
]
