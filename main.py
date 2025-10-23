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
from system_monitor import ResourceMonitor, get_project_size, get_library_sizes, get_cache_size, get_log_size

# ایجاد مدیر داده و موتور فنی
data_manager = CompleteCoinStatsManager()
technical_engine = CompleteTechnicalEngine()
signal_predictor = TradingSignalPredictor()
monitor = ResourceMonitor()

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
            prices = []
            timestamps = []
            
            for item in chart_data['result']:
                if 'price' in item and 'timestamp' in item:
                    try:
                        prices.append(float(item['price']))
                        timestamps.append(item['timestamp'])
                    except (ValueError, TypeError):
                        continue
            
            if len(prices) < 2:
                raise HTTPException(status_code=400, detail="داده‌های کافی نیست")
            
            # محاسبه اندیکاتورهای تکنیکال از داده‌های واقعی
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
                "period": period,
                "prices": prices,
                "timestamps": timestamps,
                "technical_indicators": {
                    "sma_20": [indicators.get('sma_20', prices[-1])] * len(prices),
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
                'fear_greed_index': {'value': 50}
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

@app.get("/api/system/resources")
async def get_system_resources():
    """دریافت مصرف واقعی منابع"""
    try:
        usage = monitor.get_system_usage()
        
        # اطلاعات پروژه
        project_size = get_project_size()
        lib_sizes = get_library_sizes()
        cache_size = get_cache_size()
        log_size = get_log_size()
        
        total_estimated_size = project_size + sum(lib_sizes.values())
        
        return {
            "success": True,
            "system_usage": usage,
            "project_info": {
                "total_size_mb": total_estimated_size,
                "code_size_mb": project_size,
                "libraries_size_mb": lib_sizes,
                "data_cache_size_mb": cache_size,
                "log_files_size_mb": log_size
            },
            "breakdown": {
                "fastapi": lib_sizes.get('fastapi', 0),
                "torch": lib_sizes.get('torch', 0),
                "numpy": lib_sizes.get('numpy', 0),
                "other_libs": sum(lib_sizes.values()) - lib_sizes.get('fastapi', 0) - lib_sizes.get('torch', 0) - lib_sizes.get('numpy', 0),
                "project_code": project_size,
                "cache_data": cache_size,
                "logs": log_size
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# در main.py - استفاده از system_monitor واقعی
from system_monitor import ResourceMonitor, get_project_size, get_library_sizes, get_cache_size, get_log_size

# ایجاد مانیتور
monitor = ResourceMonitor()

@app.get("/api/system/resources")
async def system_resources_real():
    """مصرف واقعی منابع سیستم"""
    try:
        usage = monitor.get_system_usage()
        
        # اطلاعات پروژه
        project_size = get_project_size()
        lib_sizes = get_library_sizes()
        cache_size = get_cache_size()
        log_size = get_log_size()
        
        total_estimated_size = project_size + sum(lib_sizes.values())
        
        return {
            "success": True,
            "system_usage": usage,
            "project_info": {
                "total_size_mb": total_estimated_size,
                "code_size_mb": project_size,
                "libraries_size_mb": lib_sizes,
                "data_cache_size_mb": cache_size,
                "log_files_size_mb": log_size
            },
            "breakdown": {
                "fastapi": lib_sizes.get('fastapi', 0),
                "torch": lib_sizes.get('torch', 0),
                "numpy": lib_sizes.get('numpy', 0),
                "other_libs": sum(lib_sizes.values()) - lib_sizes.get('fastapi', 0) - lib_sizes.get('torch', 0) - lib_sizes.get('numpy', 0),
                "project_code": project_size,
                "cache_data": cache_size,
                "logs": log_size
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "system_usage": {
                "memory": {"used_mb": 0, "percent": 0},
                "cpu": {"process_percent": 0, "system_percent": 0},
                "disk": {"used_gb": 0, "total_gb": 0, "percent": 0}
            }
        }
        
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
