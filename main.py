from fastapi import FastAPI
from system_routes import router as system_router
# از سایر routerهای خودتان import کنید

# ایجاد اپلیکیشن اصلی
app = FastAPI(
    title="Crypto AI Trading API",
    description="Advanced Cryptocurrency Analysis and Trading System",
    version="1.0.0"
)

# اضافه کردن routes سیستم
app.include_router(system_router, prefix="/api/v1", tags=["system"])

# اینجا سایر routerهای شما اضافه شوند:
 from ai_routes import router as ai_router
# app.include_router(ai_router, prefix="/api/v1", tags=["ai"])
# 
# from market_routes import router as market_router  
# app.include_router(market_router, prefix="/api/v1", tags=["market"])
# 
# from trading_routes import router as trading_router
# app.include_router(trading_router, prefix="/api/v1", tags=["trading"])

@app.get("/")
def root():
    return {
        "message": "🚀 Crypto AI Trading API is Running",
        "status": "success",
        "docs": "/docs",
        "health": "/api/v1/health/detailed"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "crypto-ai-api"}
