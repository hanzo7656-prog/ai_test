# server.py - روت‌های مخصوص فرانت‌اند
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Crypto AI Frontend")

# CORS برای توسعه
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# سرویس فایل‌های استاتیک
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_index():
    """سرویس فایل اصلی"""
    return FileResponse("index.html")

@app.get("/ai/scan")
async def serve_ai_scan():
    return FileResponse("index.html")

@app.get("/ai/technical")
async def serve_ai_technical():
    return FileResponse("index.html")

@app.get("/ai/analysis")
async def serve_ai_analysis():
    return FileResponse("index.html")

@app.get("/ai/quick")
async def serve_ai_quick():
    return FileResponse("index.html")

@app.get("/system/dashboard")
async def serve_system_dashboard():
    return FileResponse("index.html")

@app.get("/system/health")
async def serve_system_health():
    return FileResponse("index.html")

@app.get("/system/alerts")
async def serve_system_alerts():
    return FileResponse("index.html")

@app.get("/system/metrics")
async def serve_system_metrics():
    return FileResponse("index.html")

@app.get("/system/tests")
async def serve_system_tests():
    return FileResponse("index.html")

@app.get("/system/logs")
async def serve_system_logs():
    return FileResponse("index.html")

@app.get("/settings/cache")
async def serve_settings_cache():
    return FileResponse("index.html")

@app.get("/settings/ai")
async def serve_settings_ai():
    return FileResponse("index.html")

@app.get("/settings/debug")
async def serve_settings_debug():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
