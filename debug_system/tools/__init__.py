"""
Debug System Tools Package
ابزارهای توسعه، تست، گزارش‌گیری و مدیریت پیشرفته کارهای پس‌زمینه
"""

import logging

logger = logging.getLogger(__name__)

# نمونه‌های خالی - در initialize پر می‌شوند
dev_tools = None
testing_tools = None
report_generator = None
background_worker = None
task_scheduler = None
background_tasks = None
resource_manager = None
recovery_manager = None
monitoring_dashboard = None

def initialize_tools_system(debug_manager_instance=None, history_manager_instance=None):
    """راه‌اندازی و ارتباط ابزارهای توسعه و سیستم کارهای پس‌زمینه"""
    try:
        # Lazy import برای جلوگیری از circular dependency
        from .dev_tools import DevTools
        from .testing_tools import TestingTools
        from .report_generator import ReportGenerator
        from .background_worker import IntelligentBackgroundWorker, background_worker
        from .background_tasks import SmartBackgroundTasks, background_tasks
        from .resource_manager import ResourceGuardian, resource_guardian
        from .time_scheduler import TimeAwareScheduler, time_scheduler
        from .recovery_system import RecoveryManager, recovery_manager
        from .monitoring_dashboard import WorkerMonitoringDashboard, monitoring_dashboard
        
        global dev_tools, testing_tools, report_generator
        global background_worker, task_scheduler, background_tasks
        global resource_manager, recovery_manager, monitoring_dashboard
        
        # ایجاد نمونه‌های اصلی با dependency injection
        if debug_manager_instance:
            dev_tools = DevTools(debug_manager_instance)
            testing_tools = TestingTools(debug_manager_instance)
            background_tasks = SmartBackgroundTasks(debug_manager_instance, history_manager_instance)
            logger.info("✅ DevTools, TestingTools and BackgroundTasks initialized")
        
        if debug_manager_instance and history_manager_instance:
            report_generator = ReportGenerator(debug_manager_instance, history_manager_instance)
            logger.info("✅ ReportGenerator initialized")
        
        # ایجاد سیستم مدیریت منابع (نیاز به dependency ندارد)
        resource_manager = ResourceGuardian(max_cpu_percent=70.0, max_memory_percent=80.0)
        resource_manager.start_monitoring()
        logger.info("✅ Resource Manager initialized and monitoring started")
        
        # ایجاد Background Worker (نیاز به dependency ندارد)
        background_worker = IntelligentBackgroundWorker(max_workers=4, max_cpu_percent=65.0)
        background_worker.start()
        logger.info("✅ Background Worker initialized and started")
        
        # ایجاد Time Scheduler
        task_scheduler = TimeAwareScheduler(resource_manager)
        task_scheduler.start_scheduling()
        logger.info("✅ Time Scheduler initialized and started")
        
        # ایجاد Recovery Manager
        recovery_manager = RecoveryManager()
        recovery_manager.start_monitoring()
        logger.info("✅ Recovery Manager initialized and monitoring started")
        
        # ایجاد Monitoring Dashboard (وابسته به سایر کامپوننت‌ها)
        monitoring_dashboard = WorkerMonitoringDashboard(
            background_worker=background_worker,
            resource_manager=resource_manager,
            time_scheduler=task_scheduler,
            recovery_manager=recovery_manager
        )
        monitoring_dashboard.start_monitoring()
        logger.info("✅ Monitoring Dashboard initialized and monitoring started")
        
        # راه‌اندازی کارهای زمان‌بندی شده پیش‌فرض
        self._setup_default_scheduled_tasks()
        
        logger.info("🎯 Debug tools system fully initialized with advanced background workers")
        
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools, 
            "report_generator": report_generator,
            "background_worker": background_worker,
            "task_scheduler": task_scheduler,
            "background_tasks": background_tasks,
            "resource_manager": resource_manager,
            "recovery_manager": recovery_manager,
            "monitoring_dashboard": monitoring_dashboard,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Tools initialization failed: {e}")
        # ایجاد stub برای جلوگیری از خطا
        class StubTools:
            def __getattr__(self, name):
                return lambda *args, **kwargs: {"error": "Tools not initialized"}
        
        class StubWorker:
            def start(self): pass
            def stop(self): pass
            def submit_task(self, *args, **kwargs): return False, "Worker not initialized"
            def get_task_status(self, task_id): return None
            def get_detailed_metrics(self): return {"error": "Worker not initialized"}
        
        class StubManager:
            def start_monitoring(self): pass
            def stop_monitoring(self): pass
            def get_recovery_status(self): return {"error": "Manager not initialized"}
        
        # مقداردهی fallback برای کامپوننت‌های اصلی
        if debug_manager_instance and not dev_tools:
            dev_tools = StubTools()
            testing_tools = StubTools()
            background_tasks = StubTools()
        
        if debug_manager_instance and history_manager_instance and not report_generator:
            report_generator = StubTools()
        
        # مقداردهی fallback برای کامپوننت‌های جدید
        if not background_worker:
            background_worker = StubWorker()
        
        if not task_scheduler:
            task_scheduler = StubTools()
        
        if not resource_manager:
            resource_manager = StubManager()
        
        if not recovery_manager:
            recovery_manager = StubManager()
        
        if not monitoring_dashboard:
            monitoring_dashboard = StubTools()
        
        return {
            "dev_tools": dev_tools,
            "testing_tools": testing_tools,
            "report_generator": report_generator,
            "background_worker": background_worker,
            "task_scheduler": task_scheduler,
            "background_tasks": background_tasks,
            "resource_manager": resource_manager,
            "recovery_manager": recovery_manager,
            "monitoring_dashboard": monitoring_dashboard,
            "status": "partial",
            "error": str(e)
        }

def _setup_default_scheduled_tasks():
    """راه‌اندازی کارهای زمان‌بندی شده پیش‌فرض"""
    try:
        global task_scheduler, background_tasks
        
        # زمان‌بندی پاک‌سازی دوره‌ای
        task_scheduler.schedule_task(
            task_id="daily_cleanup",
            task_func=background_tasks.cleanup_temporary_files,
            task_type="light",
            interval_seconds=86400,  # هر 24 ساعت
            preferred_times=["02:00"]  # ساعت 2 بامداد
        )
        
        # زمان‌بندی گزارش روزانه
        task_scheduler.schedule_task(
            task_id="daily_analytics",
            task_func=background_tasks.generate_daily_analytics,
            task_type="light", 
            interval_seconds=86400,
            preferred_times=["03:00"]  # ساعت 3 بامداد
        )
        
        # زمان‌بندی بهینه‌سازی هفتگی (فقط آخر هفته)
        task_scheduler.schedule_task(
            task_id="weekly_optimization",
            task_func=background_tasks.run_database_optimization,
            task_type="heavy",
            interval_seconds=604800,  # هر هفته
            preferred_times=["saturday_01:00"]  # شنبه ساعت 1 بامداد
        )
        
        logger.info("📅 Default scheduled tasks configured")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to setup default scheduled tasks: {e}")
def shutdown_tools_system():
    """خاموش کردن ایمن تمام سیستم‌ها"""
    try:
        logger.info("🛑 Shutting down tools system...")
        
        # توقف مانیتورینگ‌ها
        if monitoring_dashboard:
            monitoring_dashboard.stop_monitoring()
        
        if resource_manager:
            resource_manager.stop_monitoring()
        
        if recovery_manager:
            recovery_manager.stop_monitoring()
        
        # توقف زمان‌بندی
        if task_scheduler:
            task_scheduler.stop_scheduling()
        
        # توقف worker
        if background_worker:
            background_worker.stop()
        
        logger.info("✅ Tools system shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Tools system shutdown failed: {e}")

# ایمپورت کلاس‌ها برای export
try:
    from .dev_tools import DevTools
    from .testing_tools import TestingTools
    from .report_generator import ReportGenerator
    from .background_worker import IntelligentBackgroundWorker
    from .background_tasks import SmartBackgroundTasks
    from .resource_manager import ResourceGuardian
    from .time_scheduler import TimeAwareScheduler
    from .recovery_system import RecoveryManager
    from .monitoring_dashboard import WorkerMonitoringDashboard
except ImportError as e:
    logger.warning(f"⚠️ Could not import tools classes: {e}")
    
    # ایجاد stub classes برای کامپوننت‌های اصلی
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
    
    # ایجاد stub classes برای کامپوننت‌های جدید
    class IntelligentBackgroundWorker:
        def __init__(self, *args, **kwargs):
            pass
        def start(self): pass
        def stop(self): pass
        def submit_task(self, *args, **kwargs): return False, "Worker not initialized"
        def get_task_status(self, task_id): return None
        def get_detailed_metrics(self): return {"error": "Worker not initialized"}
    
    class SmartBackgroundTasks:
        def __init__(self, debug_manager=None, history_manager=None):
            self.debug_manager = debug_manager
            self.history_manager = history_manager
    
    class ResourceGuardian:
        def __init__(self, *args, **kwargs):
            pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def get_detailed_resource_report(self): return {"error": "Resource manager not initialized"}
    
    class TimeAwareScheduler:
        def __init__(self, resource_manager=None):
            self.resource_manager = resource_manager
        def start_scheduling(self): pass
        def stop_scheduling(self): pass
        def schedule_task(self, *args, **kwargs): return False, "Scheduler not initialized"
        def get_scheduling_analytics(self): return {"error": "Scheduler not initialized"}
    
    class RecoveryManager:
        def __init__(self, *args, **kwargs):
            pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def get_recovery_status(self): return {"error": "Recovery manager not initialized"}
    
    class WorkerMonitoringDashboard:
        def __init__(self, **kwargs):
            pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
        def get_dashboard_data(self): return {"error": "Dashboard not initialized"}

# نمونه‌های گلوبال (برای دسترسی مستقیم)
try:
    from .background_worker import background_worker
    from .background_tasks import background_tasks
    from .resource_manager import resource_guardian as resource_manager
    from .time_scheduler import time_scheduler as task_scheduler
    from .recovery_system import recovery_manager
    from .monitoring_dashboard import monitoring_dashboard
except ImportError:
    # استفاده از stub instances
    background_worker = IntelligentBackgroundWorker()
    background_tasks = SmartBackgroundTasks()
    resource_manager = ResourceGuardian()
    task_scheduler = TimeAwareScheduler()
    recovery_manager = RecoveryManager()
    monitoring_dashboard = WorkerMonitoringDashboard()

# سیستم کامل tools
tools_system = {
    # کامپوننت‌های اصلی
    "dev_tools": dev_tools,
    "testing_tools": testing_tools, 
    "report_generator": report_generator,
    
    # سیستم کارهای پس‌زمینه
    "background_worker": background_worker,
    "task_scheduler": task_scheduler,
    "background_tasks": background_tasks,
    
    # مدیریت منابع و مانیتورینگ
    "resource_manager": resource_manager,
    "recovery_manager": recovery_manager,
    "monitoring_dashboard": monitoring_dashboard,
    
    # توابع مدیریت
    "initialize": initialize_tools_system,
    "shutdown": shutdown_tools_system
}

__all__ = [
    # کلاس‌های اصلی
    "DevTools", "dev_tools",
    "TestingTools", "testing_tools", 
    "ReportGenerator", "report_generator",
    
    # کلاس‌های سیستم پس‌زمینه
    "IntelligentBackgroundWorker", "background_worker",
    "SmartBackgroundTasks", "background_tasks",
    "ResourceGuardian", "resource_manager", 
    "TimeAwareScheduler", "task_scheduler",
    "RecoveryManager", "recovery_manager",
    "WorkerMonitoringDashboard", "monitoring_dashboard",
    
    # توابع مدیریت
    "initialize_tools_system", "shutdown_tools_system",
    "tools_system"
]
