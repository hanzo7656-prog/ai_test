import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class JSONFormatter(logging.Formatter):
    """فرمت‌کننده JSON برای لاگ‌های ساختاریافته"""
    
    def format(self, record: logging.LogRecord) -> str:
        """فرمت کردن رکورد لاگ به JSON"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # اضافه کردن extra fields اگر وجود دارند
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # اضافه کردن exception info اگر وجود دارد
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

class ColoredFormatter(logging.Formatter):
    """فرمت‌کننده رنگی برای کنسول"""
    
    # کدهای رنگ ANSI
    COLORS = {
        'DEBUG': '\033[94m',      # آبی
        'INFO': '\033[92m',       # سبز
        'WARNING': '\033[93m',    # زرد
        'ERROR': '\033[91m',      # قرمز
        'CRITICAL': '\033[41m'    # پس‌زمینه قرمز
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """فرمت کردن رکورد لاگ با رنگ"""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        
        # آیکون‌های مختلف برای سطوح مختلف
        icons = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '💥'
        }
        
        icon = icons.get(record.levelname, '📝')
        
        formatted_time = self.formatTime(record, self.datefmt)
        base_format = f"{log_color}{icon} [{formatted_time}] {record.levelname:8} {record.name}:{record.funcName}:{record.lineno}{self.RESET} - {record.getMessage()}"
        
        if record.exc_info:
            base_format += f"\n{log_color}Stack Trace:{self.RESET}\n{self.formatException(record.exc_info)}"
        
        return base_format

def setup_logging(
    log_dir: str = "./logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    json_logs: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """راه‌اندازی سیستم لاگ‌گیری"""
    
    # ایجاد پوشه لاگ‌ها
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # تنظیم root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # پاک کردن existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler - Rotating
    log_file = log_path / "vortexai.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, file_level.upper()))
    
    if json_logs:
        file_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error File Handler - فقط خطاها
    error_file = log_path / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # لاگ راه‌اندازی
    logging.info(f"✅ Logging system initialized - Console: {console_level}, File: {file_level}")

def get_logger(name: str, extra_fields: Dict[str, Any] = None) -> logging.Logger:
    """دریافت logger با فیلدهای اضافی"""
    logger = logging.getLogger(name)
    
    if extra_fields:
        # اضافه کردن فیلدهای اضافی به تمام رکوردهای این logger
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra_fields = extra_fields
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger

# راه‌اندازی اولیه لاگ‌گیری
setup_logging()
