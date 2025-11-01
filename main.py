from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from system_health_debug import router as system_router
from ai_analysis_routes import router as ai_router
from lbank_websocket import router as websocket_router
import logging
import os
from datetime import datetime

# ایجاد اپلیکیشن اصلی
app = FastAPI(
    title="Crypto AI Trading API",
    description="Advanced Cryptocurrency Analysis and Trading System with Sparse Neural Network",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ایجاد پوشه‌های مورد نیاز
os.makedirs("templates", exist_ok=True)
os.makedirs("templates/components", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# تنظیمات templating و static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# تنظیمات logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# اضافه کردن routes سیستم
app.include_router(system_router, prefix="/api/system", tags=["system"])
app.include_router(ai_router, prefix="/api/ai", tags=["ai-analysis"])
app.include_router(websocket_router, prefix="/api/websocket", tags=["websocket"])

# ============================ روت‌های HTML جدید ============================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """صفحه اصلی داشبورد"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health", response_class=HTMLResponse)
async def health_page(request: Request):
    """صفحه سلامت سیستم"""
    return templates.TemplateResponse("health.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    """صفحه تحلیل تکنیکال"""
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    """صفحه اسکن بازار"""
    return templates.TemplateResponse("scan.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """صفحه تنظیمات کاربر"""
    return templates.TemplateResponse("settings.html", {"request": request})

# ============================ روت‌های API اصلی ============================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy", 
        "service": "crypto-ai-api",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0"
    }

@app.get("/api/status")
def api_status():
    return {
        "api": "running",
        "websocket": "connected",
        "ai_model": "active",
        "technical_engine": "ready",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/info")
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

# ============================ روت‌های کمکی ============================

@app.get("/api/")
def root_api():
    return {
        "message": "🚀 Crypto AI Trading API is Running",
        "status": "success",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "dashboard": "/",
            "health": "/health",
            "analysis": "/analysis", 
            "scan": "/scan",
            "settings": "/settings",
            "api_docs": "/api/docs",
            "api_health": "/api/health"
        },
        "features": [
            "Real-time WebSocket Data",
            "AI Technical Analysis", 
            "Sparse Neural Network",
            "Market Scanning",
            "Advanced Indicators"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Crypto AI Trading API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
