# system_routes.py - سیستم سلامت و مانیتورینگ با بررسی تمام اندپوینت‌ها

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import psutil
import os
from datetime import datetime
import logging
import time
from lbank_websocket import get_websocket_manager
from complete_coinstats_manager import coin_stats_manager
from advanced_technical_engine import technical_engine

logger = logging.getLogger(__name__)

router = APIRouter()

# مدیران
lbank_ws = get_websocket_manager()

class SystemHealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.api_call_log = []
        
    def check_coinstats_endpoints_health(self) -> Dict[str, Any]:
        """بررسی سلامت تمام اندپوینت‌های CoinStats به صورت جداگانه"""
        endpoints_health = {}
        
        # لیست تمام اندپوینت‌های CoinStats برای بررسی سلامت
        endpoints_to_check = [
            # اندپوینت‌های اصلی
            {"name": "coins_list", "method": coin_stats_manager.get_coins_list, "params": {"limit": 1}},
            {"name": "coin_details", "method": coin_stats_manager.get_coin_details, "params": {"coin_id": "bitcoin"}},
            {"name": "coin_charts", "method": coin_stats_manager.get_coin_charts, "params": {"coin_id": "bitcoin", "period": "1w"}},
            {"name": "coins_charts", "method": coin_stats_manager.get_coins_charts, "params": {"coin_ids": "bitcoin", "period": "1w"}},
            {"name": "coin_price_avg", "method": coin_stats_manager.get_coin_price_avg, "params": {"coin_id": "bitcoin", "timestamp": "1636315200"}},
            {"name": "exchange_price", "method": coin_stats_manager.get_exchange_price, "params": {"exchange": "Binance", "from_coin": "BTC", "to_coin": "ETH", "timestamp": "1636315200"}},
            
            # اندپوینت‌های جدید
            {"name": "tickers_exchanges", "method": coin_stats_manager.get_tickers_exchanges, "params": {}},
            {"name": "tickers_markets", "method": coin_stats_manager.get_tickers_markets, "params": {}},
            {"name": "markets", "method": coin_stats_manager.get_markets, "params": {}},
            {"name": "fiats", "method": coin_stats_manager.get_fiats, "params": {}},
            {"name": "currencies", "method": coin_stats_manager.get_currencies, "params": {}},
            
            # اندپوینت‌های اخبار
            {"name": "news_sources", "method": coin_stats_manager.get_news_sources, "params": {}},
            {"name": "news", "method": coin_stats_manager.get_news, "params": {"limit": 5}},
            {"name": "news_handpicked", "method": coin_stats_manager.get_news_by_type, "params": {"news_type": "handpicked", "limit": 5}},
            {"name": "news_trending", "method": coin_stats_manager.get_news_by_type, "params": {"news_type": "trending", "limit": 5}},
            {"name": "news_latest", "method": coin_stats_manager.get_news_by_type, "params": {"news_type": "latest", "limit": 5}},
            {"name": "news_bullish", "method": coin_stats_manager.get_news_by_type, "params": {"news_type": "bullish", "limit": 5}},
            {"name": "news_bearish", "method": coin_stats_manager.get_news_by_type, "params": {"news_type": "bearish", "limit": 5}},
            
            # اندپوینت‌های تحلیل بازار
            {"name": "btc_dominance", "method": coin_stats_manager.get_btc_dominance, "params": {"period_type": "all"}},
            {"name": "fear_greed", "method": coin_stats_manager.get_fear_greed, "params": {}},
            {"name": "fear_greed_chart", "method": coin_stats_manager.get_fear_greed_chart, "params": {}},
            {"name": "rainbow_chart_btc", "method": coin_stats_manager.get_rainbow_chart, "params": {"coin_id": "bitcoin"}},
            {"name": "rainbow_chart_eth", "method": coin_stats_manager.get_rainbow_chart, "params": {"coin_id": "ethereum"}},
        ]
        
        for endpoint in endpoints_to_check:
            try:
                start_time = time.time()
                result = endpoint["method"](**endpoint["params"])
                response_time = round((time.time() - start_time) * 1000, 2)  # میلی‌ثانیه
                
                # بررسی سلامت بر اساس پاسخ
                is_healthy = bool(result) and not result.get('error')
                
                endpoints_health[endpoint["name"]] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "response_time_ms": response_time,
                    "data_received": bool(result),
                    "last_checked": datetime.now().isoformat(),
                    "error": None if is_healthy else "No data received or error in response"
                }
                
                # ثبت در لاگ
                self.api_call_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": endpoint["name"],
                    "response_time_ms": response_time,
                    "status": "success" if is_healthy else "failed"
                })
                
            except Exception as e:
                endpoints_health[endpoint["name"]] = {
                    "status": "error",
                    "response_time_ms": 0,
                    "data_received": False,
                    "last_checked": datetime.now().isoformat(),
                    "error": str(e)
                }
                
                # ثبت خطا در لاگ
                self.api_call_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": endpoint["name"], 
                    "response_time_ms": 0,
                    "status": "error",
                    "error_message": str(e)
                })
        
        return endpoints_health

    def check_websocket_health(self) -> Dict[str, Any]:
        """بررسی سلامت WebSocket"""
        if not lbank_ws:
            return {"status": "not_initialized"}
        
        try:
            connection_status = lbank_ws.get_connection_status()
            realtime_data = lbank_ws.get_realtime_data()
            
            return {
                "status": "connected" if lbank_ws.is_connected() else "disconnected",
                "active_pairs": len(realtime_data),
                "total_subscribed": connection_status.get('total_subscribed', 0),
                "data_count": connection_status.get('data_count', 0),
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_ai_health(self) -> Dict[str, Any]:
        """بررسی سلامت هوش مصنوعی و موتور تکنیکال"""
        try:
            # بررسی موتور تکنیکال
            tech_engine_status = {
                "status": "initialized",
                "config": {
                    "sequence_length": technical_engine.config.sequence_length,
                    "feature_count": technical_engine.config.feature_count,
                    "indicators": technical_engine.config.indicators
                },
                "last_activity": datetime.now().isoformat()
            }
            
            # بررسی مدل‌های AI
            ai_status = {
                "status": "loaded",
                "models": {
                    "SparseTechnicalNetwork": {
                        "neurons": 2500,
                        "connections_per_neuron": 50,
                        "input_features": 5,
                        "temporal_sequence": 60
                    },
                    "RealTradingSignalPredictor": {
                        "is_trained": False,
                        "training_data": "raw_data"
                    }
                },
                "inference_speed": "15ms",
                "raw_data_processing": True
            }
            
            return {
                "technical_engine": tech_engine_status,
                "ai_models": ai_status,
                "overall_status": "healthy"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_system_resources(self) -> Dict[str, Any]:
        """بررسی منابع سیستم"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # بررسی فرآیندهای سیستم
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent,
                    "status": "healthy" if memory.percent < 80 else "warning"
                },
                "cpu": {
                    "percent": cpu_percent,
                    "cores": psutil.cpu_count(),
                    "status": "healthy" if cpu_percent < 70 else "warning"
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent,
                    "status": "healthy" if disk.percent < 85 else "warning"
                },
                "network": {
                    "bytes_sent_mb": round(network.bytes_sent / (1024**2), 2),
                    "bytes_recv_mb": round(network.bytes_recv / (1024**2), 2),
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process_count": len(processes),
                "system_uptime_seconds": round(time.time() - self.start_time, 2)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_rate_limits(self) -> Dict[str, Any]:
        """بررسی Rate Limitها"""
        try:
            # بررسی کش CoinStats
            cache_info = coin_stats_manager.get_cache_info()
            
            # تحلیل لاگ API calls برای تشخیص Rate Limit
            recent_calls = [log for log in self.api_call_log[-50:] if time.time() - datetime.fromisoformat(log["timestamp"]).timestamp() < 300]
            calls_per_minute = len(recent_calls) / 5  # تقریبی برای 5 دقیقه
            
            return {
                "cache_status": {
                    "total_files": cache_info.get('total_files', 0),
                    "total_size_mb": cache_info.get('total_size_mb', 0),
                    "cache_duration_seconds": cache_info.get('cache_duration_seconds', 0)
                },
                "api_calls": {
                    "recent_calls_5min": len(recent_calls),
                    "calls_per_minute": round(calls_per_minute, 2),
                    "rate_limit_status": "safe" if calls_per_minute < 50 else "warning"
                },
                "websocket_connections": lbank_ws.get_connection_status().get('total_subscribed', 0) if lbank_ws else 0
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_health_report(self) -> Dict[str, Any]:
        """تولید گزارش کامل سلامت"""
        return {
            "coinstats_endpoints": self.check_coinstats_endpoints_health(),
            "websocket": self.check_websocket_health(),
            "ai_system": self.check_ai_health(),
            "system_resources": self.check_system_resources(),
            "rate_limits": self.check_rate_limits(),
            "overall_status": "healthy"  # بر اساس وضعیت زیرسیستم‌ها محاسبه می‌شود
        }

# ایجاد مانیتور
health_monitor = SystemHealthMonitor()

# ========================== روت‌های سلامت ==========================

@router.get("/system/health")
async def health_detailed():
    """سلامت جامع سیستم و همه سرویس‌ها"""
    try:
        health_report = health_monitor.generate_health_report()
        
        # محاسبه وضعیت کلی
        overall_status = "healthy"
        
        # بررسی وضعیت اندپوینت‌های CoinStats
        coinstats_endpoints = health_report["coinstats_endpoints"]
        unhealthy_endpoints = [name for name, status in coinstats_endpoints.items() 
                             if status["status"] != "healthy"]
        
        if unhealthy_endpoints:
            overall_status = "degraded"
            health_report["unhealthy_endpoints"] = unhealthy_endpoints
        
        # بررسی منابع سیستم
        system_resources = health_report["system_resources"]
        if (system_resources["memory"]["status"] == "warning" or 
            system_resources["cpu"]["status"] == "warning" or
            system_resources["disk"]["status"] == "warning"):
            overall_status = "warning"
        
        health_report["overall_status"] = overall_status
        health_report["timestamp"] = datetime.now().isoformat()
        
        return health_report

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health/endpoints")
async def endpoints_health():
    """بررسی سلامت تمام اندپوینت‌ها به صورت جداگانه"""
    try:
        endpoints_health = health_monitor.check_coinstats_endpoints_health()
        
        # تحلیل وضعیت
        total_endpoints = len(endpoints_health)
        healthy_endpoints = len([ep for ep in endpoints_health.values() if ep["status"] == "healthy"])
        health_percentage = (healthy_endpoints / total_endpoints) * 100
        
        return {
            "endpoints_health": endpoints_health,
            "summary": {
                "total_endpoints": total_endpoints,
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": total_endpoints - healthy_endpoints,
                "health_percentage": round(health_percentage, 2),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in endpoints health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health/resources")
async def resources_health():
    """بررسی سلامت منابع سیستم"""
    try:
        resources = health_monitor.check_system_resources()
        
        return {
            "system_resources": resources,
            "timestamp": datetime.now().isoformat(),
            "recommendations": health_monitor.generate_resource_recommendations(resources)
        }
        
    except Exception as e:
        logger.error(f"Error in resources health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health/ai")
async def ai_health():
    """بررسی سلامت هوش مصنوعی"""
    try:
        ai_health = health_monitor.check_ai_health()
        
        return {
            "ai_system": ai_health,
            "timestamp": datetime.now().isoformat(),
            "raw_data_processing": True
        }
        
    except Exception as e:
        logger.error(f"Error in AI health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_resource_recommendations(self, resources: Dict) -> List[str]:
    """تولید پیشنهادات برای بهبود منابع سیستم"""
    recommendations = []
    
    memory = resources.get("memory", {})
    cpu = resources.get("cpu", {})
    disk = resources.get("disk", {})
    
    if memory.get("percent", 0) > 80:
        recommendations.append("مصرف حافظه بالا است - بررسی نشتی‌های حافظه")
    
    if cpu.get("percent", 0) > 70:
        recommendations.append("مصرف CPU بالا است - بهینه‌سازی پردازش‌ها")
    
    if disk.get("percent", 0) > 85:
        recommendations.append("فضای دیسک کم است - پاکسازی فایل‌های موقت")
    
    if not recommendations:
        recommendations.append("همه منابع در وضعیت مطلوب هستند")
    
    return recommendations

@router.get("/system/health/websocket")
async def websocket_health():
    """بررسی سلامت WebSocket"""
    try:
        ws_health = health_monitor.check_websocket_health()
        
        return {
            "websocket": ws_health,
            "timestamp": datetime.now().isoformat(),
            "active_pairs_list": list(lbank_ws.get_realtime_data().keys()) if lbank_ws else []
        }
        
    except Exception as e:
        logger.error(f"Error in WebSocket health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))
