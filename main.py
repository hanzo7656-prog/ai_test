# main.py - نسخه ساده و مطمئن برای رندر
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
import os
import uvicorn

app = FastAPI()

# روت‌های API
@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head>
            <title>CryptoAI API</title>
            <meta http-equiv="refresh" content="0; url=/index.html">
        </head>
        <body>
            <p>Redirecting to CryptoAI Interface...</p>
        </body>
    </html>
    """)

@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": "crypto-ai-api", 
        "timestamp": "2024-01-01T10:00:00Z",
        "version": "3.0.0"
    })

@app.post("/api/ai/scan")
async def ai_scan():
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
                    "reasoning": "روند صعودی قوی با حجم بالا"
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
            },
            {
                "symbol": "SOL",
                "current_price": 102.45,
                "price": 102.45,
                "change": 5.67,
                "volume": "800M",
                "market_cap": "42B",
                "ai_signal": {
                    "primary_signal": "BUY", 
                    "signal_confidence": 0.81,
                    "reasoning": "شکست مقاومت کلیدی"
                }
            }
        ],
        "timestamp": "2024-01-01T10:00:00Z",
        "total_scanned": 3,
        "symbols_found": 3
    })

@app.get("/api/system/status")
async def system_status():
    return JSONResponse({
        "status": "running",
        "timestamp": "2024-01-01T10:00:00Z",
        "version": "3.0.0",
        "system_health": {
            "status": "healthy",
            "health_score": 96,
            "active_alerts": 0,
            "performance": "optimal"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting CryptoAI Server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        access_log=True
    )
