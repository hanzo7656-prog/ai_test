from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import time
import psutil
from pathlib import Path
import json
import asyncio
import logging
import sys

# ==================== DEBUG CODE ====================
print("=" * 60)
print("🛠️  VORTEXAI DEBUG - SYSTEM INITIALIZATION")
print("=" * 60)

# بررسی فایل‌ها
current_dir = os.getcwd()
print(f"📁 Current directory: {current_dir}")
print("📁 Listing files in routes directory:")
routes_dir = os.path.join(current_dir, 'routes')
if os.path.exists(routes_dir):
    for file in os.listdir(routes_dir):
        print(f"   📄 {file}")

# ایمپورت روت‌ها
try:
    from routes.health import health_router
    print("✅ health_router imported successfully!")
except ImportError as e:
    print(f"❌ Health router import error: {e}")

try:
    from complete_coinstats_manager import coin_stats_manager
    print("✅ coin_stats_manager imported successfully!")
    COINSTATS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CoinStats import error: {e}")
    COINSTATS_AVAILABLE = False

print("=" * 60)
# ==================== پایان کد دیباگ ====================

# تنظیمات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VortexAI API", 
    version="4.0.0",
    description="Complete Crypto AI System with Advanced Debugging",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ثبت روت‌ها
app.include_router(health_router)

# روت‌های اصلی (موقت تا فایل‌های دیگر ساخته شوند)
@app.get("/")
async def root():
    """صفحه اصلی"""
    return {
        "message": "VortexAI API Server v4.0.0", 
        "version": "4.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "debug_system": "active",
        "endpoints": {
            "health": "/api/health/status",
            "docs": "/docs",
            "debug_dashboard": "/api/health/debug/endpoints"
        }
    }

# روت‌های موقت برای تست
@app.get("/api/coins/{symbol}")
async def get_coin_temp(symbol: str):
    """روت موقت کوین‌ها"""
    return {
        "symbol": symbol,
        "price": 45000,
        "status": "temp_endpoint",
        "debug": {
            "response_time": "15ms",
            "cache_used": False
        }
    }

@app.get("/api/news")
async def get_news_temp():
    """روت موقت اخبار"""
    return {
        "news": ["temp_news_1", "temp_news_2"],
        "status": "temp_endpoint"
    }

# مدیریت خطای 404
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": {
                "health": "/api/health/status",
                "coins": "/api/coins/{symbol}",
                "news": "/api/news",
                "docs": "/docs"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Starting VortexAI Server v4.0.0 on port {port}")
    print(f"🔧 Debug System: ACTIVE")
    print(f"📊 Health Dashboard: http://localhost:{port}/api/health/status")
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)
