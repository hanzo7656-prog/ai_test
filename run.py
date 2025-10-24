#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
🎯 AI Trading Assistant - Complete Version v3.0
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware  # 🔽 این خط را اضافه کن

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# ایجاد WebSocket Manager اولیه
lbank_ws = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت طول عمر برنامه"""
    # Startup
    logger.info("🚀 Starting AI Trading Assistant v3.0...")
    
    # راه‌اندازی WebSocket
    global lbank_ws
    try:
        from lbank_websocket import LBankWebSocketManager
        lbank_ws = LBankWebSocketManager()
        logger.info("✅ LBank WebSocket Initialized - Auto-connecting...")
        
        # منتظر اتصال اولیه WebSocket
        await asyncio.sleep(3)
        
    except Exception as e:
        logger.error(f"❌ Error initializing WebSocket: {e}")
        lbank_ws = None
    
    # راه‌اندازی سرویس‌های دیگر
    try:
        from complete_routes import set_websocket_manager
        if lbank_ws:
            set_websocket_manager(lbank_ws)
            logger.info("✅ WebSocket manager set for routes")
    except Exception as e:
        logger.error(f"❌ Error setting up routes: {e}")
    
    logger.info("✅ All services initialized")
    
    yield  # برنامه اجرا می‌شود
    
    # Shutdown
    logger.info("🛑 Shutting down AI Trading Assistant...")
    if lbank_ws:
        try:
            lbank_ws.disconnect()
            logger.info("✅ WebSocket disconnected")
        except Exception as e:
            logger.error(f"❌ Error disconnecting WebSocket: {e}")

# ایجاد اپلیکیشن FastAPI
app = FastAPI(
    title="AI Trading Assistant",
    description="سیستم کامل تحلیل بازار با هوش مصنوعی",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔽 MIDDLEWARE جدید را اینجا اضافه کن - دقیقاً بعد از CORS
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"📥 {request.method} {request.url}")
        try:
            response = await call_next(request)
            logger.info(f"📤 {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "message": str(e)}
            )

# اضافه کردن middleware به اپ
app.add_middleware(LoggingMiddleware)

# 🔽 Exception handlers را اینجا اضافه کن - بعد از middleware
@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"💥 Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "خطای داخلی سرور رخ داده است",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found", 
            "message": "آدرس درخواستی یافت نشد",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"💥 Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "خطای غیرمنتظره رخ داده است",
            "timestamp": datetime.now().isoformat()
        }
    )

# سرو کردن فایل‌های استاتیک
app.mount("/static", StaticFiles(directory="static"), name="static")

# ایمپورت routeها
try:
    from complete_routes import router as main_router
    from ai_analysis_routes import router as ai_router
    
    # ثبت routeها
    app.include_router(main_router)
    app.include_router(ai_router)
    logger.info("✅ All routes registered successfully")
    
except Exception as e:
    logger.error(f"❌ Error importing routes: {e}")

# route اصلی
@app.get("/")
async def root():
    """صفحه اصلی API"""
    websocket_status = "disconnected"
    active_pairs = 0
    
    if lbank_ws:
        websocket_status = "connected" if lbank_ws.is_connected() else "disconnected"
        active_pairs = len(lbank_ws.get_realtime_data())
    
    return {
        "message": "AI Trading Assistant API", 
        "version": "3.0.0",
        "status": "running",
        "websocket_status": websocket_status,
        "active_pairs": active_pairs,
        "endpoints": {
            "health": "/health",
            "websocket_status": "/websocket/status", 
            "market_data": "/market/overview",
            "ai_analysis": "/ai/analysis",
            "system_resources": "/api/system/resources"
        }
    }

@app.get("/health")
async def health_check():
    """بررسی سلامت سرویس"""
    websocket_connected = False
    active_pairs = 0
    
    if lbank_ws:
        websocket_connected = lbank_ws.is_connected()
        active_pairs = len(lbank_ws.get_realtime_data())
    
    return {
        "status": "healthy",
        "timestamp": int(datetime.now().timestamp()),
        "services": {
            "api": "running",
            "websocket": "connected" if websocket_connected else "disconnected",
            "data_service": "ready"
        },
        "metrics": {
            "active_websocket_pairs": active_pairs
        }
    }

@app.get("/websocket/status")
async def websocket_status():
    """وضعیت WebSocket"""
    if not lbank_ws:
        return {
            "connected": False,
            "error": "WebSocket not initialized"
        }
    
    return {
        "connected": lbank_ws.is_connected(),
        "active_pairs": list(lbank_ws.realtime_data.keys()),
        "data_count": len(lbank_ws.realtime_data)
    }

@app.get("/websocket/data/{symbol}")
async def get_websocket_data(symbol: str):
    """دریافت داده لحظه‌ای WebSocket"""
    if not lbank_ws:
        return {
            "error": "WebSocket not initialized",
            "symbol": symbol
        }
    
    data = lbank_ws.get_realtime_data(symbol)
    if not data:
        return {
            "error": "Symbol not found in WebSocket data",
            "symbol": symbol
        }
    
    return {
        "symbol": symbol,
        "data": data
    }

async def main():
    """تابع اصلی"""
    try:
        import uvicorn
        
        # ایجاد پوشه‌های لازم
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("coinstats_collected_data", exist_ok=True)
        os.makedirs("raw_data", exist_ok=True)
        
        # دریافت پورت از محیط
        port = int(os.environ.get("PORT", 8000))
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        logger.info(f"🌐 Server starting on port {port}")
        
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        raise

if __name__ == "__main__":  
    try:
        # بررسی نسخه پایتون
        logger.info(f"🐍 Python version: {sys.version}")
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
