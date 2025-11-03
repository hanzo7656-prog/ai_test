# main.py - بدون importهای مشکل‌دار
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ایجاد پوشه frontend
os.makedirs("frontend", exist_ok=True)

# ==================== APIهای اصلی ====================

@app.get("/api/health")
def health_check():
    return JSONResponse({
        "status": "healthy", 
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "service": "crypto-ai"
    })

@app.post("/api/ai/scan")
def ai_scan():
    return JSONResponse({
        "status": "success",
        "scan_results": [
            {
                "symbol": "BTC",
                "current_price": 45231.50,
                "price": 45231.50,
                "change": 2.34,
                "volume": "2.5B",
                "market_cap": "886B",
                "ai_signal": {
                    "primary_signal": "BUY",
                    "signal_confidence": 0.87,
                    "reasoning": "روند صعودی قوی"
                }
            },
            {
                "symbol": "ETH",
                "current_price": 2534.20,
                "price": 2534.20,
                "change": -0.89,
                "volume": "1.3B", 
                "market_cap": "304B",
                "ai_signal": {
                    "primary_signal": "HOLD",
                    "signal_confidence": 0.73,
                    "reasoning": "ثبات در کانال قیمتی"
                }
            }
        ],
        "timestamp": datetime.now().isoformat(),
        "total_scanned": 2,
        "symbols_found": 2
    })

@app.get("/api/system/status")
def system_status():
    return JSONResponse({
        "status": "running",
        "timestamp": datetime.now().isoformat(), 
        "version": "3.0.0",
        "system_health": {
            "status": "healthy",
            "health_score": 95
        }
    })

@app.get("/api/system/alerts")
def system_alerts():
    return JSONResponse({
        "status": "success",
        "alerts": [
            {
                "id": "alert_1", 
                "title": "سیستم فعال است",
                "message": "همه چیز خوب کار می‌کند",
                "level": "info",
                "timestamp": datetime.now().isoformat()
            }
        ],
        "total_alerts": 1
    })

@app.get("/api/info")
def system_info():
    return JSONResponse({
        "name": "Crypto AI Trading System",
        "version": "3.0.0",
        "status": "running", 
        "timestamp": datetime.now().isoformat()
    })

# ==================== فرانت‌اند ====================

@app.get("/")
def serve_frontend():
    try:
        return FileResponse("frontend/index.html")
    except:
        return JSONResponse({"error": "Frontend not found"}, status_code=404)

@app.get("/{full_path:path}")
def serve_all_routes(full_path: str):
    # اگر مسیر API نیست، فرانت‌اند رو برگردون
    if not full_path.startswith('api/'):
        try:
            return FileResponse("frontend/index.html")
        except:
            return JSONResponse({"error": "Page not found"}, status_code=404)
    else:
        return JSONResponse({"error": "API endpoint not found"}, status_code=404)

# هندل خطا
@app.exception_handler(404)
def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
)
