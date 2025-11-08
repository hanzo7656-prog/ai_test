from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import time
import psutil
from pathlib import Path
import json
import asyncio
import logging
import sys

# ==================== DEBUG CODE ====================
print("=" * 60)
print("🛠️  VORTEXAI DEBUG - SYSTEM INITIALIZATION")
print("=" * 60)

# ایمپورت روت‌ها
try:
    from routes.health import health_router
    from routes.coins import coins_router
    from routes.exchanges import exchanges_router
    from routes.news import news_router
    from routes.insights import insights_router
    from routes.raw_coins import raw_coins_router
    from routes.raw_news import raw_news_router
    from routes.raw_insights import raw_insights_router
    from routes.docs import docs_router
    print("✅ All routers imported successfully!")
except ImportError as e:
    print(f"❌ Router import error: {e}")

try:
    from complete_coinstats_manager import coin_stats_manager
    print("✅ coin_stats_manager imported successfully!")
    COINSTATS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CoinStats import error: {e}")
    COINSTATS_AVAILABLE = False

# ==================== DEBUG SYSTEM IMPORTS ====================
try:
    from debug_system.core import core_system, debug_manager, metrics_collector, alert_manager
    from debug_system.monitors import monitors_system, endpoint_monitor, system_monitor, performance_monitor, security_monitor
    from debug_system.storage import history_manager, log_manager, cache_debugger
    from debug_system.realtime import websocket_manager, console_stream
    from debug_system.tools import tools_system, dev_tools, testing_tools, report_generator
    
    # ایمپورت جداگانه LiveDashboardManager
    from debug_system.realtime.live_dashboard import LiveDashboardManager
    
    DEBUG_SYSTEM_AVAILABLE = True
    print("✅ Complete debug system imported successfully!")
except ImportError as e:
    print(f"❌ Debug system import error: {e}")
    DEBUG_SYSTEM_AVAILABLE = False

print("=" * 60)
# ==================== پایان کد دیباگ ====================

# تنظیمات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VortexAI API", 
    version="4.0.0",
    description="Complete Crypto AI System with Advanced Debugging",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DEBUG SYSTEM INITIALIZATION ====================
live_dashboard_manager = None
console_stream_manager = None

if DEBUG_SYSTEM_AVAILABLE:
    try:
        # راه‌اندازی کامل سیستم دیباگ
        print("🔄 Initializing debug system...")
        
        # راه‌اندازی هسته (اگر قبلاً راه‌اندازی نشده)
        if not core_system:
            from debug_system.core import initialize_core_system
            core_system = initialize_core_system()
        
        # راه‌اندازی مانیتورها (اگر قبلاً راه‌اندازی نشده)
        if not monitors_system:
            from debug_system.monitors import initialize_monitors_system
            monitors_system = initialize_monitors_system()
        
        # تکمیل راه‌اندازی ابزارها با endpoint_monitor
        if not tools_system:
            from debug_system.tools import initialize_tools_system
            tools_system = initialize_tools_system(monitors_system["endpoint_monitor"])
        
        # راه‌اندازی سیستم real-time
        live_dashboard_manager = LiveDashboardManager(
            debug_manager, 
            metrics_collector
        )
        
        # شروع برودکست دشبورد
        asyncio.create_task(live_dashboard_manager.start_dashboard_broadcast())
        
        # تنظیم console stream
        console_stream_manager = console_stream.ConsoleStreamManager()
        
        print("✅ Complete debug system initialized and activated!")
        print(f"   - Core Modules: {len(core_system) if core_system else 0}")
        print(f"   - Monitors: {len(monitors_system) if monitors_system else 0}")
        print(f"   - Tools: {len(tools_system) if tools_system else 0}")
        print(f"   - Real-time Systems: Active")
        
    except Exception as e:
        print(f"❌ Debug system initialization error: {e}")
        import traceback
        traceback.print_exc()
        DEBUG_SYSTEM_AVAILABLE = False
else:
    print("❌ Debug system is not available")

# ثبت روت‌ها
app.include_router(health_router)
app.include_router(coins_router)
app.include_router(exchanges_router)
app.include_router(news_router)
app.include_router(insights_router)
app.include_router(raw_coins_router)
app.include_router(raw_news_router)
app.include_router(raw_insights_router)
app.include_router(docs_router)

# ==================== DEBUG ROUTES ====================
if DEBUG_SYSTEM_AVAILABLE and live_dashboard_manager and console_stream_manager:
    @app.get("/debug/dashboard")
    async def debug_dashboard():
        """صفحه دشبورد دیباگ"""
        return FileResponse("debug_system/realtime/templates/dashboard.html")
    
    @app.get("/debug/console")
    async def debug_console():
        """صفحه کنسول دیباگ"""
        return FileResponse("debug_system/realtime/templates/console.html")
    
    @app.websocket("/debug/ws/dashboard")
    async def websocket_dashboard(websocket: WebSocket):
        """WebSocket برای دشبورد real-time"""
        await live_dashboard_manager.connect_dashboard(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            live_dashboard_manager.disconnect_dashboard(websocket)
    
    @app.websocket("/debug/ws/console")
    async def websocket_console(websocket: WebSocket):
        """WebSocket برای کنسول real-time"""
        await console_stream_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            console_stream_manager.disconnect(websocket)

# ==================== 🗺️ ROADMAP COMPLETE - راهنمای کامل روت‌ها ====================

VORTEXAI_ROADMAP = {
    "project": "VortexAI API v4.0.0",
    "description": "Complete Crypto AI System with 9 Main Routes",
    "version": "4.0.0",
    "timestamp": datetime.now().isoformat(),
    
    "🚀 MAIN ROUTES": {
        "description": "۸ روت مادر اصلی سیستم",
        "routes": {
            # 1. سلامت سیستم
            "HEALTH": {
                "base_path": "/api/health",
                "description": "سلامت و مانیتورینگ سیستم",
                "endpoints": {
                    "status": "GET /api/health/status - وضعیت کلی سیستم",
                    "overview": "GET /api/health/overview - نمای کلی سیستم",
                    "ping": "GET /api/health/ping - تست حیات سیستم",
                    "version": "GET /api/health/version - نسخه‌های سیستم",
                    "debug_endpoints": "GET /api/health/debug/endpoints - دیباگ اندپوینت‌ها",
                    "debug_system": "GET /api/health/debug/system - دیباگ سیستم",
                    "debug_reports_daily": "GET /api/health/debug/reports/daily - گزارش روزانه",
                    "debug_reports_performance": "GET /api/health/debug/reports/performance - گزارش عملکرد",
                    "debug_reports_security": "GET /api/health/debug/reports/security - گزارش امنیتی",
                    "debug_metrics_live": "GET /api/health/debug/metrics/live - متریک‌های زنده"
                }
            },
            
            # 2. نمادها (پردازش شده)
            "COINS": {
                "base_path": "/api/coins",
                "description": "داده‌های پردازش شده نمادها",
                "endpoints": {
                    "list": "GET /api/coins/list - لیست نمادها",
                    "details": "GET /api/coins/details/{coin_id} - جزئیات نماد",
                    "charts": "GET /api/coins/charts/{coin_id} - چارت نماد", 
                    "multi_charts": "GET /api/coins/multi-charts - چارت چندنماد",
                    "price_avg": "GET /api/coins/price/avg - قیمت متوسط"
                }
            },
            
            # 3. صرافی‌ها (پردازش شده)
            "EXCHANGES": {
                "base_path": "/api/exchanges", 
                "description": "داده‌های پردازش شده صرافی‌ها",
                "endpoints": {
                    "list": "GET /api/exchanges/list - لیست صرافی‌ها",
                    "markets": "GET /api/exchanges/markets - مارکت‌ها",
                    "fiats": "GET /api/exchanges/fiats - ارزهای فیات",
                    "currencies": "GET /api/exchanges/currencies - ارزها",
                    "price": "GET /api/exchanges/price - قیمت صرافی"
                }
            },
            
            # 4. اخبار (پردازش شده)
            "NEWS": {
                "base_path": "/api/news",
                "description": "اخبار و تحلیل‌های پردازش شده", 
                "endpoints": {
                    "all": "GET /api/news/all - اخبار عمومی",
                    "by_type": "GET /api/news/type/{news_type} - اخبار بر اساس نوع",
                    "sources": "GET /api/news/sources - منابع خبری",
                    "detail": "GET /api/news/detail/{news_id} - جزئیات خبر"
                }
            },
            
            # 5. بینش و تحلیل (پردازش شده)
            "INSIGHTS": {
                "base_path": "/api/insights",
                "description": "تحلیل‌های بازار و بینش‌ها",
                "endpoints": {
                    "btc_dominance": "GET /api/insights/btc-dominance - دامیننس بیت‌کوین",
                    "fear_greed": "GET /api/insights/fear-greed - شاخص ترس و طمع",
                    "fear_greed_chart": "GET /api/insights/fear-greed/chart - چارت ترس و طمع",
                    "rainbow_chart": "GET /api/insights/rainbow-chart/{coin_id} - چارت رنگین‌کمان"
                }
            },
            
            # 6. داده‌های خام نمادها
            "RAW_COINS": {
                "base_path": "/api/raw/coins", 
                "description": "داده‌های خام نمادها - برای هوش مصنوعی",
                "endpoints": {
                    "list": "GET /api/raw/coins/list - لیست خام نمادها",
                    "details": "GET /api/raw/coins/details/{coin_id} - جزئیات خام نماد",
                    "charts": "GET /api/raw/coins/charts/{coin_id} - چارت خام نماد",
                    "multi_charts": "GET /api/raw/coins/multi-charts - چارت خام چندنماد",
                    "price_avg": "GET /api/raw/coins/price/avg - قیمت متوسط خام",
                    "exchange_price": "GET /api/raw/coins/price/exchange - قیمت صرافی خام",
                    "metadata": "GET /api/raw/coins/metadata - متادیتای نمادها",
                    "filters": "GET /api/raw/coins/filters - فیلترهای موجود"
                }
            },
            
            # 7. داده‌های خام اخبار
            "RAW_NEWS": {
                "base_path": "/api/raw/news",
                "description": "داده‌های خام اخبار - برای هوش مصنوعی",
                "endpoints": {
                    "all": "GET /api/raw/news/all - اخبار عمومی خام", 
                    "by_type": "GET /api/raw/news/type/{news_type} - اخبار خام بر اساس نوع",
                    "sources": "GET /api/raw/news/sources - منابع خبری خام",
                    "detail": "GET /api/raw/news/detail/{news_id} - جزئیات خبر خام",
                    "sentiment_analysis": "GET /api/raw/news/sentiment-analysis - تحلیل احساسات",
                    "metadata": "GET /api/raw/news/metadata - متادیتای اخبار"
                }
            },
            
            # 8. داده‌های خام بینش
            "RAW_INSIGHTS": {
                "base_path": "/api/raw/insights",
                "description": "داده‌های خام بینش و تحلیل - برای هوش مصنوعی",
                "endpoints": {
                    "btc_dominance": "GET /api/raw/insights/btc-dominance - دامیننس بیت‌کوین خام",
                    "fear_greed": "GET /api/raw/insights/fear-greed - شاخص ترس و طمع خام", 
                    "fear_greed_chart": "GET /api/raw/insights/fear-greed/chart - چارت ترس و طمع خام",
                    "rainbow_chart": "GET /api/raw/insights/rainbow-chart/{coin_id} - چارت رنگین‌کمان خام",
                    "market_analysis": "GET /api/raw/insights/market-analysis - تحلیل جامع بازار",
                    "metadata": "GET /api/raw/insights/metadata - متادیتای بینش‌ها"
                }
            }
        }
    },
    
    "📚 DOCUMENTATION": {
        "description": "مستندات کامل و مثال‌های کاربردی",
        "routes": {
            "complete_docs": "GET /api/docs/complete - مستندات کامل API",
            "coins_docs": "GET /api/docs/coins - مستندات تخصصی نمادها", 
            "code_examples": "GET /api/docs/examples - مثال‌های کد",
            "interactive_docs": "GET /docs - مستندات تعاملی (Swagger UI)",
            "redoc_docs": "GET /redoc - مستندات زیبا (ReDoc)"
        }
    },
    
    "🔧 DEBUG & MONITORING": {
        "description": "سیستم دیباگ و مانیتورینگ پیشرفته",
        "routes": {
            "DEBUG_DASHBOARD": "GET /debug/dashboard - دشبورد دیباگ",
            "DEBUG_CONSOLE": "GET /debug/console - کنسول دیباگ",
            "DEBUG_WS_DASHBOARD": "WS /debug/ws/dashboard - WebSocket دشبورد",
            "DEBUG_WS_CONSOLE": "WS /debug/ws/console - WebSocket کنسول",
            "METRICS_ALL": "GET /api/health/metrics - تمام متریک‌ها",
            "ALERTS_ACTIVE": "GET /api/health/alerts - هشدارهای فعال",
            "REPORTS_DAILY": "GET /api/health/reports/daily - گزارش روزانه",
            "REALTIME_CONSOLE": "WS /api/health/debug/realtime/console - کنسول Real-Time",
            "REALTIME_DASHBOARD": "WS /api/health/debug/realtime/dashboard - دشبورد Real-Time"
        }
    },
    
    "🛠️ DEVELOPER TOOLS": {
        "description": "ابزارهای توسعه و تست",
        "routes": {
            "TEST_TRAFFIC": "POST /api/health/tools/test-traffic - تولید ترافیک تست",
            "LOAD_TEST": "POST /api/health/tools/load-test - تست بار", 
            "DEPENDENCIES": "GET /api/health/tools/dependencies - بررسی وابستگی‌ها",
            "MEMORY_ANALYSIS": "GET /api/health/tools/memory-analysis - آنالیز حافظه"
        }
    },
    
    "📊 QUICK ACCESS EXAMPLES": {
        "description": "دسترسی سریع به اندپوینت‌های مهم",
        "examples": {
            "HEALTH_CHECK": "/api/health/status",
            "BITCOIN_DETAILS": "/api/coins/details/bitcoin", 
            "BITCOIN_RAW": "/api/raw/coins/details/bitcoin",
            "COINS_LIST": "/api/coins/list?limit=10",
            "FEAR_GREED": "/api/insights/fear-greed",
            "LATEST_NEWS": "/api/news/all?limit=5",
            "EXCHANGES_LIST": "/api/exchanges/list",
            "SYSTEM_METRICS": "/api/health/metrics/system",
            "DEBUG_ENDPOINTS": "/api/health/debug/endpoints",
            "DEBUG_SYSTEM": "/api/health/debug/system",
            "COMPLETE_DOCS": "/api/docs/complete",
            "CODE_EXAMPLES": "/api/docs/examples",
            "AI_DATA_SAMPLES": "/api/raw/coins/metadata"
        }
    },
    
    "🎯 USAGE PATTERNS": {
        "frontend_basic": "برای فرانت‌اند: استفاده از روت‌های پردازش شده (/api/coins/, /api/news/)",
        "frontend_advanced": "برای نمودارها: استفاده از روت‌های خام (/api/raw/coins/charts/)", 
        "mobile_app": "برای موبایل: روت‌های پردازش شده + سلامت سیستم",
        "ai_analysis": "برای هوش مصنوعی: روت‌های خام + بینش‌ها",
        "admin_panel": "برای ادمین: تمام روت‌های سلامت و دیباگ",
        "external_integration": "برای یکپارچه‌سازی: روت‌های خام + وضعیت سیستم",
        "new_developers": "برای توسعه‌دهندگان جدید: شروع با /api/docs/complete و /api/roadmap"
    },
    
    "⚡ PERFORMANCE TIPS": {
        "use_processed": "برای نمایش عمومی از روت‌های پردازش شده استفاده کنید (سریع‌تر)",
        "use_raw": "برای تحلیل‌های پیشرفته از روت‌های خام استفاده کنید (داده کامل)",
        "caching": "داده‌ها به مدت ۵ دقیقه کش می‌شوند",
        "pagination": "برای لیست‌های بزرگ از صفحه‌بندی استفاده کنید",
        "health_check": "قبل از درخواست‌های مهم سلامت سیستم را بررسی کنید"
    },
    
    "🤖 AI TRAINING DATA": {
        "description": "داده‌های مناسب برای آموزش هوش مصنوعی",
        "raw_coins_data": "/api/raw/coins/list?limit=1000",
        "raw_news_sentiment": "/api/raw/news/sentiment-analysis",
        "market_insights": "/api/raw/insights/market-analysis", 
        "historical_charts": "/api/raw/coins/charts/bitcoin?period=all",
        "metadata_structure": "/api/raw/coins/metadata"
    }
}

@app.get("/")
async def root():
    """صفحه اصلی با راهنمای کامل روت‌ها"""
    return {
        "message": "🚀 VortexAI API Server v4.0.0 - Complete Crypto AI System",
        "version": "4.0.0", 
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc", 
            "roadmap": "/api/roadmap",
            "complete_docs": "/api/docs/complete",
            "code_examples": "/api/docs/examples"
        },
        "quick_start": {
            "health_check": "/api/health/status",
            "bitcoin_data": "/api/coins/details/bitcoin",
            "latest_news": "/api/news/all?limit=5",
            "market_sentiment": "/api/insights/fear-greed",
            "ai_data_samples": "/api/raw/coins/metadata",
            "debug_endpoints": "/api/health/debug/endpoints",
            "debug_system": "/api/health/debug/system"
        },
        "system_info": {
            "total_routes": len(app.routes),
            "debug_system": "active" if DEBUG_SYSTEM_AVAILABLE else "inactive",
            "coinstats_available": COINSTATS_AVAILABLE,
            "startup_time": datetime.now().isoformat(),
            "ai_ready": True
        }
    }

@app.get("/api/roadmap")
async def get_roadmap():
    """دریافت راهنمای کامل روت‌های سیستم"""
    return VORTEXAI_ROADMAP

@app.get("/api/quick-reference")
async def quick_reference():
    """مرجع سریع روت‌های مهم"""
    return {
        "title": "VortexAI API - Quick Reference",
        "description": "مرجع سریع برای دسترسی به اندپوینت‌های اصلی",
        "timestamp": datetime.now().isoformat(),
        
        "essential_endpoints": {
            "health": {
                "url": "/api/health/status",
                "description": "بررسی سلامت سیستم"
            },
            "coins_list": {
                "url": "/api/coins/list", 
                "description": "لیست نمادها"
            },
            "coin_details": {
                "url": "/api/coins/details/{coin_id}",
                "description": "جزئیات نماد خاص"
            },
            "coin_charts": {
                "url": "/api/coins/charts/{coin_id}",
                "description": "داده‌های چارت"
            },
            "news": {
                "url": "/api/news/all",
                "description": "اخبار بازار"
            },
            "fear_greed": {
                "url": "/api/insights/fear-greed",
                "description": "شاخص ترس و طمع"
            },
            "exchanges": {
                "url": "/api/exchanges/list",
                "description": "لیست صرافی‌ها"
            }
        },
        
        "debug_endpoints": {
            "debug_endpoints": {
                "url": "/api/health/debug/endpoints",
                "description": "وضعیت دیباگ اندپوینت‌ها"
            },
            "debug_system": {
                "url": "/api/health/debug/system",
                "description": "وضعیت کامل سیستم دیباگ"
            },
            "debug_dashboard": {
                "url": "/debug/dashboard",
                "description": "دشبورد دیباگ real-time"
            },
            "debug_console": {
                "url": "/debug/console",
                "description": "کنسول دیباگ real-time"
            },
            "daily_report": {
                "url": "/api/health/debug/reports/daily",
                "description": "گزارش روزانه دیباگ"
            },
            "live_metrics": {
                "url": "/api/health/debug/metrics/live",
                "description": "متریک‌های زنده"
            }
        },
        
        "ai_data_endpoints": {
            "raw_coins": {
                "url": "/api/raw/coins/details/{coin_id}",
                "description": "داده‌های خام نماد برای AI"
            },
            "raw_charts": {
                "url": "/api/raw/coins/charts/{coin_id}", 
                "description": "داده‌های خام چارت برای AI"
            },
            "raw_news": {
                "url": "/api/raw/news/all",
                "description": "اخبار خام برای AI"
            },
            "sentiment_analysis": {
                "url": "/api/raw/news/sentiment-analysis",
                "description": "تحلیل احساسات برای AI"
            },
            "market_analysis": {
                "url": "/api/raw/insights/market-analysis",
                "description": "تحلیل بازار برای AI"
            }
        },
        
        "documentation": {
            "complete_docs": {
                "url": "/api/docs/complete",
                "description": "مستندات کامل API"
            },
            "code_examples": {
                "url": "/api/docs/examples", 
                "description": "مثال‌های کد"
            },
            "interactive_docs": {
                "url": "/docs",
                "description": "مستندات تعاملی"
            }
        }
    }

@app.get("/api/endpoints/count")
async def count_endpoints():
    """شمردن تعداد کل اندپوینت‌ها"""
    total_endpoints = 0
    routes_info = []
    
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            total_endpoints += len(route.methods)
            routes_info.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, "name", "Unknown")
            })
    
    return {
        "total_endpoints": total_endpoints,
        "total_routes": len(app.routes),
        "timestamp": datetime.now().isoformat(),
        "routes_by_category": {
            "health": len([r for r in routes_info if '/api/health' in r['path']]),
            "coins": len([r for r in routes_info if '/api/coins' in r['path']]),
            "raw_coins": len([r for r in routes_info if '/api/raw/coins' in r['path']]),
            "news": len([r for r in routes_info if '/api/news' in r['path']]),
            "raw_news": len([r for r in routes_info if '/api/raw/news' in r['path']]),
            "insights": len([r for r in routes_info if '/api/insights' in r['path']]),
            "raw_insights": len([r for r in routes_info if '/api/raw/insights' in r['path']]),
            "exchanges": len([r for r in routes_info if '/api/exchanges' in r['path']]),
            "documentation": len([r for r in routes_info if '/api/docs' in r['path']]),
            "debug": len([r for r in routes_info if '/debug' in r['path']])
        },
        "sample_routes": routes_info[:10]  # نمایش ۱۰ تا اول
    }

@app.get("/api/system/info")
async def system_info():
    """اطلاعات کامل سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "server_time": datetime.now().isoformat(),
            "uptime_seconds": int(time.time() - psutil.boot_time())
        },
        "resources": {
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2)
        },
        "api_status": {
            "total_endpoints": len(app.routes),
            "coinstats_available": COINSTATS_AVAILABLE,
            "debug_system_available": DEBUG_SYSTEM_AVAILABLE,
            "debug_system_status": "active" if DEBUG_SYSTEM_AVAILABLE else "inactive",
            "version": "4.0.0",
            "ai_ready": True
        },
        "timestamp": datetime.now().isoformat()
    }

# مدیریت خطای 404
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "timestamp": datetime.now().isoformat(),
            "suggestions": {
                "check_docs": "Visit /api/docs/complete for complete documentation",
                "check_roadmap": "Visit /api/roadmap for system overview", 
                "check_health": "Visit /api/health/status to check system health",
                "common_endpoints": {
                    "health": "/api/health/status",
                    "coins_list": "/api/coins/list", 
                    "news": "/api/news/all",
                    "insights": "/api/insights/fear-greed",
                    "ai_data": "/api/raw/coins/metadata",
                    "debug_endpoints": "/api/health/debug/endpoints"
                }
            },
            "quick_links": {
                "interactive_docs": "/docs",
                "quick_reference": "/api/quick-reference", 
                "system_info": "/api/system/info"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    
    print("🚀" * 50)
    print("🎯 VORTEXAI API SERVER v4.0.0 - AI READY")
    print("🚀" * 50)
    print(f"📊 Total Routes: {len(app.routes)}")
    print(f"🌐 Server URL: http://localhost:{port}")
    print(f"📚 Documentation: http://localhost:{port}/docs")
    print(f"🗺️  Roadmap: http://localhost:{port}/api/roadmap")
    print(f"📖 Complete Docs: http://localhost:{port}/api/docs/complete")
    print("🎯 Quick Start:")
    print(f"   • Health Check: http://localhost:{port}/api/health/status")
    print(f"   • Bitcoin Details: http://localhost:{port}/api/coins/details/bitcoin") 
    print(f"   • Latest News: http://localhost:{port}/api/news/all?limit=5")
    print(f"   • Fear & Greed: http://localhost:{port}/api/insights/fear-greed")
    print(f"   • AI Data Samples: http://localhost:{port}/api/raw/coins/metadata")
    print(f"   • Debug Endpoints: http://localhost:{port}/api/health/debug/endpoints")
    print(f"   • Debug System: http://localhost:{port}/api/health/debug/system")
    print("🔧 Debug System: " + ("✅ FULLY ACTIVE" if DEBUG_SYSTEM_AVAILABLE else "❌ UNAVAILABLE"))
    if DEBUG_SYSTEM_AVAILABLE:
        print(f"   • Real-time Dashboard: http://localhost:{port}/debug/dashboard")
        print(f"   • Debug Console: http://localhost:{port}/debug/console")
        print(f"   • System Reports: http://localhost:{port}/api/health/debug/reports/daily")
    print("🤖 AI Ready: ✅ YES")
    print("📈 CoinStats API: " + ("✅ AVAILABLE" if COINSTATS_AVAILABLE else "❌ UNAVAILABLE"))
    print("🚀" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)
