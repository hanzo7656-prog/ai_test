# routes/ml_routes/__init__.py
"""
روت‌های هوش مصنوعی - ML Routes
ماژول مدیریت کامل اندپوینت‌های هوش مصنوعی

انواع اندپوینت‌ها:
1. 📊 آنالیز و مانیتورینگ (ml_analysis.py)
2. 🎯 پیش‌بینی‌های مدل (predictions.py) 
3. 🎓 مدیریت آموزش (training.py)

Integration با سیستم اصلی:
- استفاده از ۳ دیتابیس Redis مخصوص AI (UTA, UTB, UTC)
- ارتباط با ۴ روت داده خام (raw_coins, raw_exchanges, raw_news, raw_insights)
- گزارش سلامت به روت سلامت مادر
"""

from .ml_analysis import ml_analysis_router
from .predictions import predictions_router
from .training import training_router

# لیست تمام روت‌ها برای ثبت در برنامه اصلی
ml_routers = [
    ml_analysis_router,
    predictions_router, 
    training_router
]

# اطلاعات ماژول برای مستندات
__version__ = "1.0.0"
__author__ = "Vortex AI System"
__description__ = "مدیریت کامل اندپوینت‌های هوش مصنوعی و یادگیری ماشینی"

# متادیتای اندپوینت‌ها برای مستندات خودکار
ENDPOINTS_METADATA = {
    "ml_analysis": {
        "prefix": "/api/ml",
        "tags": ["ML Analysis"],
        "description": "آنالیز و مانیتورینگ سیستم هوش مصنوعی",
        "endpoints": [
            {"path": "/health", "method": "GET", "desc": "سلامت سیستم AI"},
            {"path": "/models", "method": "GET", "desc": "لیست مدل‌های فعال"},
            {"path": "/analyze/market", "method": "POST", "desc": "تحلیل جامع بازار"},
            {"path": "/performance/metrics", "method": "GET", "desc": "متریک‌های عملکرد"},
            {"path": "/data/quality", "method": "GET", "desc": "گزارش کیفیت داده‌ها"},
            {"path": "/features/engineered", "method": "GET", "desc": "ویژگی‌های مهندسی شده"},
            {"path": "/alerts/active", "method": "GET", "desc": "هشدارهای فعال"}
        ]
    },
    "predictions": {
        "prefix": "/api/ml/predict", 
        "tags": ["ML Predictions"],
        "description": "پیش‌بینی‌های مدل‌های هوش مصنوعی",
        "endpoints": [
            {"path": "/technical/{model_name}", "method": "POST", "desc": "پیش‌بینی تحلیل تکنیکال"},
            {"path": "/batch/technical", "method": "POST", "desc": "پیش‌بینی دسته‌ای"},
            {"path": "/confidence/{model_name}", "method": "GET", "desc": "سطح اطمینان مدل"},
            {"path": "/market/sentiment", "method": "POST", "desc": "پیش‌بینی احساسات بازار"},
            {"path": "/custom/{model_name}", "method": "POST", "desc": "پیش‌بینی سفارشی"}
        ]
    },
    "training": {
        "prefix": "/api/ml/training",
        "tags": ["ML Training"], 
        "description": "مدیریت آموزش و ارزیابی مدل‌ها",
        "endpoints": [
            {"path": "/start/{model_name}", "method": "POST", "desc": "شروع آموزش مدل"},
            {"path": "/start/autonomous", "method": "POST", "desc": "شروع آموزش خودکار"},
            {"path": "/status/{model_name}", "method": "GET", "desc": "وضعیت آموزش مدل"},
            {"path": "/status", "method": "GET", "desc": "وضعیت تمام آموزش‌ها"},
            {"path": "/stop/{model_name}", "method": "POST", "desc": "توقف آموزش مدل"},
            {"path": "/history", "method": "GET", "desc": "تاریخچه آموزش"},
            {"path": "/schedule/{model_name}", "method": "POST", "desc": "زمان‌بندی آموزش"},
            {"path": "/evaluate/{model_name}", "method": "POST", "desc": "ارزیابی مدل"}
        ]
    }
}

def get_ml_routes_info():
    """دریافت اطلاعات کامل درباره روت‌های هوش مصنوعی"""
    return {
        "module": "ml_routes",
        "version": __version__,
        "description": __description__,
        "total_routers": len(ml_routers),
        "total_endpoints": sum(len(meta["endpoints"]) for meta in ENDPOINTS_METADATA.values()),
        "routers": [
            {
                "name": router_name,
                "prefix": meta["prefix"],
                "tags": meta["tags"],
                "description": meta["description"],
                "endpoints_count": len(meta["endpoints"]),
                "endpoints": meta["endpoints"]
            }
            for router_name, meta in ENDPOINTS_METADATA.items()
        ],
        "dependencies": {
            "databases": ["UTA_REDIS_AI", "UTB_REDIS_AI", "UTC_REDIS_AI"],
            "data_sources": ["raw_coins", "raw_exchanges", "raw_news", "raw_insights"],
            "core_modules": ["ml_core", "self_learning", "data_pipeline"]
        }
    }

def initialize_ml_routes():
    """مقداردهی اولیه روت‌های هوش مصنوعی"""
    try:
        # بررسی وجود ماژول‌های وابسته
        from ml_core import initialize_ml_core
        from self_learning import autonomous_trainer
        from data_pipeline import feature_engineer, data_validator
        
        # مقداردهی اولیه هسته ML
        ml_core_components = initialize_ml_core()
        
        print("✅ ML Routes initialized successfully!")
        print(f"   - Available routers: {len(ml_routers)}")
        print(f"   - Total endpoints: {get_ml_routes_info()['total_endpoints']}")
        print(f"   - ML Core: {len(ml_core_components['model_manager'].active_models)} models")
        print(f"   - Data Pipeline: Ready")
        print(f"   - Self Learning: Ready")
        
        return {
            "routers": ml_routers,
            "info": get_ml_routes_info(),
            "components": ml_core_components
        }
        
    except ImportError as e:
        print(f"⚠️ ML Routes dependencies not fully available: {e}")
        return {
            "routers": ml_routers,
            "info": get_ml_routes_info(),
            "warning": "Some dependencies not available"
        }
    except Exception as e:
        print(f"❌ Error initializing ML Routes: {e}")
        raise

# اجرای خودکار مقداردهی اولیه هنگام ایمپورت
try:
    ml_routes_initialized = initialize_ml_routes()
    print("🚀 ML Routes are ready to use!")
except Exception as e:
    print(f"⚠️ ML Routes auto-initialization skipped: {e}")
    ml_routes_initialized = {"routers": ml_routers, "info": get_ml_routes_info()}

__all__ = [
    'ml_analysis_router',
    'predictions_router',
    'training_router', 
    'ml_routers',
    'get_ml_routes_info',
    'initialize_ml_routes',
    'ENDPOINTS_METADATA'
]
