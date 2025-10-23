from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime
import asyncio

app = FastAPI(title="AI Trading Dashboard")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# سرویس‌های موجود
from complete_coinstats_manager import CompleteCoinStatsManager
from technical_engine_complete import CompleteTechnicalEngine
from ultra_efficient_trading_transformer import TradingSignalPredictor

# ایجاد مدیر داده و موتور فنی
data_manager = CompleteCoinStatsManager()
technical_engine = CompleteTechnicalEngine()
signal_predictor = TradingSignalPredictor()

# منتظر اتصال WebSocket
async def wait_for_websocket():
    """منتظر اتصال WebSocket می‌ماند"""
    for i in range(30):  # 30 ثانیه منتظر می‌ماند
        if data_manager.ws_connected and data_manager.realtime_data:
            return True
        await asyncio.sleep(1)
    return False

@app.get("/")
async def serve_dashboard():
    """سرو کردن صفحه اصلی دشبورد"""
    return FileResponse('templates/dashboard.html')

@app.get("/chart/data/{symbol}")
async def get_chart_data(symbol: str, period: str = "1d"):
    """دریافت داده‌های نمودار واقعی"""
    try:
        # دریافت داده‌های واقعی
        chart_data = data_manager.get_coin_charts(symbol, period)
        
        if not chart_data:
            raise HTTPException(status_code=404, detail="داده‌های چارت یافت نشد")
        
        # پردازش داده‌های واقعی برای نمودار
        if isinstance(chart_data, dict) and 'result' in chart_data:
            prices = [float(item['price']) for item in chart_data['result'] if 'price' in item]
            timestamps = [item['timestamp'] for item in chart_data['result'] if 'timestamp' in item]
            
            # محاسبه اندیکاتورهای تکنیکال از داده‌های واقعی
            if len(prices) >= 20:
                ohlc_data = {
                    'open': prices[:-1],
                    'high': [max(prices[i], prices[i+1]) for i in range(len(prices)-1)],
                    'low': [min(prices[i], prices[i+1]) for i in range(len(prices)-1)],
                    'close': prices[1:],
                    'volume': [1000000] * len(prices)  # حجم نمونه
                }
                
                indicators = technical_engine.calculate_all_indicators(ohlc_data)
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "period": period,
                    "prices": prices,
                    "timestamps": timestamps,
                    "technical_indicators": {
                        "sma_20": [indicators.get('sma_20', 0)] * len(prices),
                        "rsi": [indicators.get('rsi', 50)] * len(prices),
                        "volume": [1000000] * len(prices)
                    }
                }
        
        raise HTTPException(status_code=404, detail="داده‌های کافی نیست")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت داده‌های چارت: {str(e)}")

@app.get("/api/signals")
async def get_trading_signals():
    """دریافت سیگنال‌های معاملاتی واقعی"""
    try:
        # منتظر داده‌های واقعی از WebSocket
        if not await wait_for_websocket():
            raise HTTPException(status_code=503, detail="WebSocket متصل نیست")
        
        # دریافت داده‌های لحظه‌ای واقعی
        btc_data = data_manager.get_realtime_price('btc_usdt')
        eth_data = data_manager.get_realtime_price('eth_usdt')
        
        if not btc_data:
            raise HTTPException(status_code=404, detail="داده‌های لحظه‌ای یافت نشد")
        
        # آماده‌سازی داده‌های بازار برای مدل AI
        market_data = {
            'price_data': {
                'historical_prices': [btc_data.get('price', 50000)],
                'volume_data': [btc_data.get('volume', 1000000)]
            },
            'technical_indicators': {
                'momentum_indicators': {
                    'rsi': technical_engine._rsi([btc_data.get('price', 50000)], 14) if btc_data.get('price') else 50
                },
                'trend_indicators': {
                    'macd': {'value': technical_engine._macd([btc_data.get('price', 50000)])[0] if btc_data.get('price') else 0}
                }
            },
            'market_data': {
                'fear_greed_index': {'value': 50}  # مقدار پیش‌فرض
            }
        }
        
        # تولید سیگنال واقعی
        result = signal_predictor.predict_signals(market_data)
        
        return {
            "success": True,
            "signals": result.get("signals", {}),
            "market_data": {
                "btc_price": btc_data.get('price', 0),
                "eth_price": eth_data.get('price', 0),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در تولید سیگنال: {str(e)}")

@app.get("/api/market/overview")
async def get_market_overview():
    """دریافت overview بازار واقعی"""
    try:
        # منتظر اتصال WebSocket
        if not await wait_for_websocket():
            raise HTTPException(status_code=503, detail="WebSocket متصل نیست")
        
        # داده‌های واقعی از WebSocket
        btc_data = data_manager.get_realtime_price('btc_usdt')
        eth_data = data_manager.get_realtime_price('eth_usdt')
        
        if not btc_data:
            raise HTTPException(status_code=404, detail="داده‌های بازار یافت نشد")
        
        # محاسبه تغییرات قیمت
        current_price = btc_data.get('price', 0)
        change_24h = btc_data.get('change', 0)
        
        return {
            "success": True,
            "current_price": current_price,
            "price_change_24h": change_24h,
            "volume_24h": btc_data.get('volume', 0),
            "high_24h": btc_data.get('high_24h', 0),
            "low_24h": btc_data.get('low_24h', 0),
            "timestamp": btc_data.get('timestamp', ''),
            "websocket_connected": data_manager.ws_connected
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت داده‌های بازار: {str(e)}")

@app.get("/api/technical/indicators")
async def get_technical_indicators(symbol: str = "bitcoin"):
    """دریافت اندیکاتورهای تکنیکال واقعی"""
    try:
        # دریافت داده‌های تاریخی
        historical_data = data_manager.get_coin_charts(symbol, "1d")
        
        if not historical_data or 'result' not in historical_data:
            raise HTTPException(status_code=404, detail="داده‌های تاریخی یافت نشد")
        
        # استخراج قیمت‌ها
        prices = [float(item['price']) for item in historical_data['result'] if 'price' in item]
        
        if len(prices) < 20:
            raise HTTPException(status_code=400, detail="داده‌های تاریخی کافی نیست")
        
        # محاسبه اندیکاتورهای واقعی
        ohlc_data = {
            'open': prices[:-1],
            'high': [max(prices[i], prices[i+1]) for i in range(len(prices)-1)],
            'low': [min(prices[i], prices[i+1]) for i in range(len(prices)-1)],
            'close': prices[1:],
            'volume': [1000000] * (len(prices) - 1)
        }
        
        indicators = technical_engine.calculate_all_indicators(ohlc_data)
        
        return {
            "success": True,
            "symbol": symbol,
            "indicators": {
                "rsi": indicators.get('rsi', 50),
                "macd": indicators.get('macd', 0),
                "sma_20": indicators.get('sma_20', 0),
                "sma_50": indicators.get('sma_50', 0),
                "bollinger_upper": indicators.get('bb_upper', 0),
                "bollinger_lower": indicators.get('bb_lower', 0),
                "stochastic_k": indicators.get('stoch_k', 50),
                "stochastic_d": indicators.get('stoch_d', 50)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در محاسبه اندیکاتورها: {str(e)}")

@app.get("/api/system/status")
async def get_system_status():
    """دریافت وضعیت سیستم"""
    try:
        # اطلاعات حافظه
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "success": True,
            "websocket_connected": data_manager.ws_connected,
            "active_pairs": len(data_manager.realtime_data),
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت وضعیت سیستم: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
