# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

# ایمپورت درست روت‌ها
from system_routes import router as system_router
from ai_analysis_routes import router as ai_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting AI Trading Assistant - Final Version")
    yield
    logger.info("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan, title="AI Trading Assistant", version="5.0")

# ثبت روت‌ها - حتماً اینطور باشه
app.include_router(system_router)
app.include_router(ai_router)

@app.get("/")
async def root():
    return {
        "message": "AI Trading Assistant API - Final Version",
        "version": "5.0.0", 
        "endpoints": {
            "health": "/health/detailed",
            "system": "/system/debug", 
            "ai_analysis": "/ai/analysis",
            "scan": "/scan/advanced",
            "technical": "/technical/analysis"
        }
    }
