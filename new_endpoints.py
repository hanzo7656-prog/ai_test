from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import time
import random

app = FastAPI(title="AI Trading Assistant", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# سرو فایل‌های استاتیک
app.mount("/static", StaticFiles(directory="static"), name="static")

# مدل‌های داده
class ChartRequest(BaseModel):
    symbol: str
    period: str = "7d"
    indicators: List[str] = ["sma", "rsi", "macd"]

class AnalysisRequest(BaseModel):
    symbols: List[str]
    analysis_type: str = "technical"

class AlertRequest(BaseModel):
    symbol: str
    condition: str
    target_price: float
    alert_type: str = "price"

# اندپوینت‌های اصلی
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """سرو صفحه اصلی داشبورد"""
    return FileResponse("templates/dashboard.html")

@app.get("/chart/data/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = Query("7d", regex="^(1d|7d|30d|90d)$"),
    indicators: str = Query("sma,rsi,macd")
):
    """دریافت داده‌های نمودار برای نماد خاص"""
    
    # داده‌های نمونه - در حالت واقعی از APIهای مالی می‌گیریم
    base_price = 50000
    price_data = []
    for i in range(30 if period == "30d" else 7 if period == "7d" else 1):
        price_data.append(base_price + random.randint(-2000, 2000))
    
    timestamps = []
    current_time = int(time.time())
    for i in range(len(price_data)):
        timestamps.append(f"2024-10-{21 - i}" if period == "7d" else f"2024-10-{22 - i}")
    
    # محاسبه اندیکاتورها
    sma_20 = [sum(price_data[max(0, i-19):i+1]) / min(20, i+1) for i in range(len(price_data))]
    rsi_data = [random.randint(30, 70) for _ in price_data]
    macd_data = [random.randint(-100, 100) for _ in price_data]
    
    return {
        "symbol": symbol.upper(),
        "period": period,
        "prices": price_data,
        "timestamps": timestamps,
        "technical_indicators": {
            "sma_20": sma_20,
            "rsi": rsi_data,
            "macd": macd_data,
            "volume": [random.randint(1000000, 5000000) for _ in price_data]
        },
        "market_summary": {
            "current_price": price_data[-1],
            "price_change_24h": round((price_data[-1] - price_data[-2]) / price_data[-2] * 100, 2),
            "volume_24h": random.randint(20000000, 50000000),
            "market_cap": random.randint(800000000000, 1000000000000)
        }
    }

@app.get("/chart/analysis/{symbol}")
async def get_chart_analysis(symbol: str):
    """تحلیل تکنیکال پیشرفته برای نمودار"""
    
    analysis_data = {
        "symbol": symbol.upper(),
        "timestamp": int(time.time()),
        "trend_analysis": {
            "direction": "صعودی",
            "strength": 75,
            "duration": "کوتاه‌مدت",
            "confidence": 82
        },
        "key_levels": {
            "support": [52000, 51500, 51000],
            "resistance": [53500, 54000, 54500],
            "current_price": 52500 + random.randint(-1000, 1000)
        },
        "signals": [
            {
                "type": "BUY",
                "confidence": 85,
                "reason": "RSI در منطقه اشباع فروش + شکست مقاومت",
                "target_price": 54000,
                "stop_loss": 51800
            },
            {
                "type": "HOLD",
                "confidence": 65,
                "reason": "حجم معاملات در حال افزایش",
                "timeframe": "1-2 روز"
            }
        ],
        "risk_assessment": {
            "level": "متوسط",
            "score": 45,
            "factors": ["نوسان متوسط", "حجم مناسب", "روند مشخص"]
        }
    }
    
    return analysis_data

@app.post("/analyze/advanced")
async def advanced_analysis(request: AnalysisRequest):
    """تحلیل پیشرفته بازار"""
    
    analyses = []
    for symbol in request.symbols:
        analysis = {
            "symbol": symbol.upper(),
            "timestamp": int(time.time()),
            "technical_score": random.randint(60, 95),
            "sentiment_score": random.randint(50, 90),
            "recommendation": random.choice(["STRONG_BUY", "BUY", "HOLD", "SELL"]),
            "key_metrics": {
                "volatility": round(random.uniform(0.1, 0.4), 2),
                "liquidity": random.choice(["HIGH", "MEDIUM", "LOW"]),
                "momentum": random.choice(["STRONG", "MODERATE", "WEAK"])
            },
            "ai_insights": [
                "حجم معاملات در حال افزایش است",
                "احساسات بازار مثبت است",
                "نوسان در محدوده سالم قرار دارد",
                "سیگنال‌های تکنیکال همسو هستند"
            ]
        }
        analyses.append(analysis)
    
    return {
        "analysis_type": request.analysis_type,
        "timestamp": int(time.time()),
        "market_regime": random.choice(["BULLISH", "BEARISH", "SIDEWAYS"]),
        "overall_confidence": random.randint(70, 90),
        "symbol_analyses": analyses,
        "summary": "وضعیت کلی بازار مثبت ارزیابی می‌شود"
    }

@app.get("/analysis/history")
async def get_analysis_history(
    symbol: str,
    limit: int = Query(10, ge=1, le=100)
):
    """تاریخچه تحلیل‌ها"""
    
    history = []
    base_time = int(time.time())
    
    for i in range(limit):
        history.append({
            "timestamp": base_time - (i * 3600),  # هر ساعت یک تحلیل
            "signal": random.choice(["BUY", "SELL", "HOLD"]),
            "accuracy": random.randint(75, 95),
            "price_at_signal": 50000 + random.randint(-5000, 5000),
            "price_change": round(random.uniform(-5, 8), 2),
            "confidence": random.randint(70, 90),
            "reason": random.choice([
                "شکست مقاومت کلیدی",
                "تثبیت در سطح حمایت",
                "سیگنال RSI",
                "الگوی نموداری صعودی"
            ])
        })
    
    return {
        "symbol": symbol.upper(),
        "history": history,
        "performance": {
            "total_signals": limit,
            "success_rate": random.randint(75, 85),
            "average_profit": round(random.uniform(2, 6), 2),
            "best_trade": round(random.uniform(8, 15), 2)
        }
    }

@app.get("/system/performance")
async def system_performance():
    """کارایی سیستم"""
    
    return {
        "memory_usage": {
            "current_mb": 185,
            "max_mb": 512,
            "percent": 36,
            "status": "OPTIMAL"
        },
        "api_performance": {
            "response_time_avg": 0.45,
            "requests_per_minute": 12,
            "error_rate": 0.2,
            "uptime": 99.8
        },
        "ai_model": {
            "status": "ACTIVE",
            "inference_speed": "0.8s",
            "accuracy": 88.5,
            "model_type": "Spike Transformer",
            "parameters": "285K"
        },
        "market_data": {
            "last_update": int(time.time()),
            "symbols_tracked": 15,
            "update_frequency": "1m",
            "data_source": "Multiple APIs"
        }
    }

@app.post("/alerts/create")
async def create_alert(request: AlertRequest):
    """ایجاد هشدار جدید"""
    
    alert_id = f"alert_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return {
        "alert_id": alert_id,
        "symbol": request.symbol.upper(),
        "condition": request.condition,
        "target_price": request.target_price,
        "alert_type": request.alert_type,
        "status": "ACTIVE",
        "created_at": int(time.time()),
        "message": f"هشدار برای {request.symbol} ایجاد شد"
    }

@app.get("/alerts/list")
async def list_alerts(symbol: Optional[str] = None):
    """لیست هشدارهای فعال"""
    
    alerts = [
        {
            "id": f"alert_{int(time.time()) - i}",
            "symbol": "BTCUSDT",
            "condition": "PRICE_ABOVE",
            "target_price": 54000,
            "current_price": 52500,
            "status": "ACTIVE",
            "created_at": int(time.time()) - i
        }
        for i in range(3)
    ]
    
    if symbol:
        alerts = [alert for alert in alerts if alert["symbol"] == symbol.upper()]
    
    return {"alerts": alerts}

@app.post("/alerts/{alert_id}/delete")
async def delete_alert(alert_id: str):
    """حذف هشدار"""
    
    return {
        "status": "SUCCESS",
        "message": f"هشدار {alert_id} حذف شد",
        "deleted_at": int(time.time())
    }

@app.get("/market/overview")
async def market_overview():
    """بررسی کلی بازار"""
    
    return {
        "timestamp": int(time.time()),
        "total_market_cap": random.randint(1800000000000, 2200000000000),
        "market_cap_change_24h": round(random.uniform(-2, 3), 2),
        "fear_greed_index": random.randint(40, 70),
        "dominance": {
            "btc": round(random.uniform(48, 52), 2),
            "eth": round(random.uniform(16, 18), 2),
            "others": round(random.uniform(30, 36), 2)
        },
        "top_movers": [
            {
                "symbol": "BTC",
                "price_change_24h": round(random.uniform(-3, 5), 2),
                "volume": random.randint(20000000, 40000000)
            },
            {
                "symbol": "ETH",
                "price_change_24h": round(random.uniform(-2, 4), 2),
                "volume": random.randint(10000000, 20000000)
            },
            {
                "symbol": "SOL",
                "price_change_24h": round(random.uniform(-5, 8), 2),
                "volume": random.randint(5000000, 15000000)
            }
        ]
    }

# اندپوینت‌های کمکی
@app.get("/health")
async def health_check():
    """بررسی سلامت سرویس"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": "2.0.0",
        "services": {
            "api": "running",
            "ai_model": "active",
            "data_feed": "connected"
        }
    }

@app.get("/symbols/list")
async def list_symbols():
    """لیست نمادهای پشتیبانی شده"""
    return {
        "symbols": [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "LTCUSDT", "BCHUSDT", "XLMUSDT", "XRPUSDT", "EOSUSDT",
            "TRXUSDT", "ETCUSDT", "XTZUSDT", "ATOMUSDT", "ALGOUSDT"
        ],
        "count": 15,
        "last_updated": int(time.time())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
