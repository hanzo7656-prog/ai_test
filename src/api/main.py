# 📁 src/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# ایجاد اپلیکیشن FastAPI
app = FastAPI(
    title="Crypto Market Analyzer API",
    description="Real-time cryptocurrency market analysis and trading signals",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در production محدود کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Crypto Market Analyzer API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "crypto-analyzer"}

# روت‌های ساده برای تست
@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working!"}

# این روت رو حذف کردم چون داده ساختگی برمیگردوند
# @app.get("/api/symbols/{symbol}")
# async def get_symbol_info(symbol: str):
#     return {"symbol": symbol, "price": 50000, "change": 2.5}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
