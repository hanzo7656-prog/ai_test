#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
🎯 AI Trading Assistant - Complete Version v3.0
با WebSocket LBank و تمام routeهای جدید
"""

import asyncio
import sys
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

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
    """مدیریت طول عمر برنامه با روش جدید FastAPI"""
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
        
        # بررسی وضعیت اتصال
        if lbank_ws and lbank_ws.is_connected():
            logger.info("✅ WebSocket connected successfully")
        else:
            logger.warning("⚠️ WebSocket not connected yet (still trying in background)")
        
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
    logger.info("📡 Available endpoints:")
    logger.info("   GET  / - صفحه اصلی")
    logger.info("   GET  /health - بررسی سلامت")
    logger.info("   GET  /websocket/status - وضعیت WebSocket")
    logger.info("   GET  /websocket/data/{symbol} - داده‌های لحظه‌ای")
    logger.info("   POST /ai/analysis - تحلیل هوش مصنوعی")
    logger.info("   GET  /news/latest - اخبار بازار")
    logger.info("   POST /alerts/create - ایجاد هشدار")
    logger.info("   GET  /alerts/list - لیست هشدارها")
    logger.info("   GET  /data/raw/{data_type} - داده‌های خام")
    logger.info("   GET  /api/system/resources - مصرف منابع سیستم")
    logger.info("   GET  /market/overview - نمای کلی بازار")
    
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
    memory_usage = "unknown"
    
    if lbank_ws:
        websocket_connected = lbank_ws.is_connected()
        active_pairs = len(lbank_ws.get_realtime_data())
    
    # بررسی مصرف حافظه
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_usage = f"{memory_mb:.1f}MB"
    except:
        memory_usage = "unavailable"
    
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {
            "api": "running",
            "websocket": "connected" if websocket_connected else "disconnected",
            "data_service": "ready"
        },
        "metrics": {
            "active_websocket_pairs": active_pairs,
            "memory_usage": memory_usage
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
    
    status = lbank_ws.get_connection_status()
    return {
        "connected": lbank_ws.is_connected(),
        "active_pairs": status.get('active_pairs', []),
        "data_count": status.get('data_count', 0),
        "subscribed_pairs": status.get('subscribed_pairs', []),
        "total_subscribed": status.get('total_subscribed', 0)
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

# هندل خطاهای عمومی
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"💥 Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": str(exc)
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "path": request.url.path
    }

async def main():
    """تابع اصلی - نسخه تعمیر شده"""
    try:
        import uvicorn
        
        # ایجاد پوشه‌های لازم
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("coinstats_collected_data", exist_ok=True)
        os.makedirs("raw_data", exist_ok=True)
        
        # 🔧 دریافت پورت از محیط - نسخه درست
        port = int(os.environ.get("PORT", 8000))  # ✅ درست شد!
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        logger.info(f"🌐 Server starting on port {port}")
        logger.info(f"📊 Access the API at: http://localhost:{port}")
        
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        raise

if __name__ == "__main__":  
    try:
        # بررسی نسخه پایتون
        logger.info(f"🐍 Python version: {sys.version}")
        
        # بررسی کتابخانه‌های نصب شده
        try:
            import fastapi
            import uvicorn
            import websocket
            import requests
            logger.info("✅ All required packages are installed")
        except ImportError as e:
            logger.error(f"❌ Missing package: {e}")
            sys.exit(1)
            
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
