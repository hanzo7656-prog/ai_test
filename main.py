from fastapi import FastAPI
from system_health_debug import router as system_router
from ai_analysis_routes import router as ai_router
from lbank_websocket import router as websocket_router
import logging
from datetime import datetime

# ایجاد اپلیکیشن اصلی
app = FastAPI(
    title="Crypto AI Trading API",
    description="Advanced Cryptocurrency Analysis and Trading System with Sparse Neural Network",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# تنظیمات logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# اضافه کردن routes سیستم
app.include_router(system_router, tags=["system"])

# اضافه کردن routes تحلیل AI
app.include_router(ai_router, tags=["ai-analysis"])

# اضافه کردن routes WebSocket
app.include_router(websocket_router, tags=["websocket"])

# Route اصلی
@app.get("/")
def root():
    return {
        "message": "🚀 Crypto AI Trading API is Running",
        "status": "success",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health/detailed",
        "features": [
            "Real-time WebSocket Data",
            "AI Technical Analysis", 
            "Sparse Neural Network",
            "Market Scanning",
            "Advanced Indicators"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "service": "crypto-ai-api",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
def api_status():
    return {
        "api": "running",
        "websocket": "connected",
        "ai_model": "active",
        "technical_engine": "ready",
        "timestamp": datetime.now().isoformat()
    }

# Route برای اطلاعات سیستم
@app.get("/info")
def system_info():
    return {
        "name": "Crypto AI Trading System",
        "version": "3.0.0",
        "architecture": "Sparse Neural Network",
        "total_neurons": 2500,
        "supported_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        "features": [
            "Real-time market data",
            "AI-powered analysis",
            "Technical indicators",
            "Pattern recognition",
            "Risk management"
        ],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Crypto AI Trading API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
