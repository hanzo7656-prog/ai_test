# 📁 src/api/main.py  (فقط API - نه موتور!)

from fastapi import FastAPI
from .routes.analysis import analysis_router
from .routes.trading import trading_router  
from .routes.system import system_router

app = FastAPI(
    title="Crypto Market Analyzer API",
    description="Real-time cryptocurrency market analysis and trading signals", 
    version="1.0.0"
)

# رجیستر کردن روت‌ها
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(trading_router, prefix="/api/v1/trading", tags=["Trading"]) 
app.include_router(system_router, prefix="/api/v1/system", tags=["System"])

@app.get("/")
async def root():
    return {"message": "Crypto Market Analyzer API", "status": "running"}
