#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI Trading Assistant - Complete Version v3.0
با تمام روت‌های جدید و WebSocket LBank
"""

import asyncio
import sys
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# تنظیمات logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# ایجاد اپلیکیشن
app = FastAPI(
    title="AI Trading Assistant",
    description="سیستم کامل تحلیل بازار با هوش مصنوعی",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ایمپورت روت‌ها
from complete_routes import router as main_router, set_websocket_manager
from ai_analysis_routes import router as ai_router
from lbank_websocket import LBankWebSocketManager

# ثبت روت‌ها
app.include_router(main_router)
app.include_router(ai_router)

# راه‌اندازی WebSocket
lbank_ws = LBankWebSocketManager()
set_websocket_manager(lbank_ws)  # ارسال به complete_routes

@app.on_event("startup")
async def startup_event():
    """رویداد راه‌اندازی"""
    logger.info("🚀 Starting AI Trading Assistant v3.0...")
    
    # راه‌اندازی WebSocket
    logger.info("🌐 Starting LBank WebSocket...")
    lbank_ws.start()
    
    logger.info("✅ All services initialized")
    logger.info("📊 Available endpoints:")
    logger.info("   POST /ai/analysis     - تحلیل هوش مصنوعی")
    logger.info("   GET  /websocket/data  - داده‌های لحظه‌ای")
    logger.info("   GET  /news/latest     - اخبار بازار")
    logger.info("   POST /alerts/create   - ایجاد هشدار")
    logger.info("   GET  /data/raw        - داده‌های خام")

@app.on_event("shutdown")
async def shutdown_event():
    """رویداد خاموشی"""
    logger.info("🛑 Shutting down AI Trading Assistant...")

async def main():
    """تابع اصلی"""
    try:
        import uvicorn
        
        # ایجاد پوشه‌های لازم
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        logger.info(f"🌐 Server starting on port {config.port}")
        await server.serve()
        
    except Exception as e:
        logger.error(f"💥 Server failed to start: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
