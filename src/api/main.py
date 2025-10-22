# 📁 main.py  (ریشه پروژه)

import asyncio
import os
import sys
from src.api.main import app

if __name__ == "__main__":
    # اجرای سرور FastAPI
    import uvicorn
    
    # ایجاد پوشه‌های لازم
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # فقط برای توسعه
        log_level="info"
    )
