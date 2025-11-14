# routes/ml_routes/ml_analysis.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ml_core import ml_model_manager, ml_health_monitor, data_integrator
from data_pipeline import feature_engineer, data_validator

logger = logging.getLogger(__name__)

ml_analysis_router = APIRouter(prefix="/api/ml", tags=["ML Analysis"])

@ml_analysis_router.get("/health")
async def get_ml_health():
    """دریافت سلامت سیستم هوش مصنوعی"""
    try:
        health_report = ml_health_monitor.get_system_health()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": health_report
        }
    except Exception as e:
        logger.error(f"❌ Error getting ML health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_analysis_router.get("/models")
async def get_models_list():
    """دریافت لیست مدل‌های فعال"""
    try:
        models_info = {}
        for model_name, model_info in ml_model_manager.active_models.items():
            models_info[model_name] = {
                "version": model_info['version'],
                "created_at": model_info['created_at'].isoformat(),
                "last_used": model_info['last_used'].isoformat(),
                "parameters": model_info['model'].config.total_neurons if hasattr(model_info['model'], 'config') else 'unknown'
            }
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_models": len(models_info),
                "models": models_info
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting models list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_analysis_router.post("/analyze/market")
async def analyze_market_data():
    """آنالیز جامع داده‌های بازار"""
    try:
        # جمع‌آوری داده‌های خام
        logger.info("🔍 Starting comprehensive market analysis...")
        
        raw_data = await data_integrator.collect_raw_data()
        
        # اعتبارسنجی داده‌ها
        validation_report = data_validator.validate_data_quality(raw_data)
        
        if validation_report['overall_quality'] == 'poor':
            logger.warning("⚠️ Poor data quality, analysis may be limited")
        
        # مهندسی ویژگی‌ها
        engineered_features = feature_engineer.engineer_market_features(raw_data)
        
        # تحلیل با مدل‌های هوش مصنوعی
        analysis_results = {}
        
        # تحلیل با مدل تحلیل‌گر تکنیکال
        if 'technical_analyzer' in ml_model_manager.active_models:
            try:
                # تبدیل ویژگی‌ها به فرمت مناسب برای مدل
                model_input = await _prepare_model_input(engineered_features)
                technical_analysis = await ml_model_manager.predict('technical_analyzer', model_input)
                analysis_results['technical_analysis'] = technical_analysis
            except Exception as e:
                logger.error(f"❌ Technical analysis failed: {e}")
                analysis_results['technical_analysis'] = {'error': str(e)}
        
        # ایجاد گزارش نهایی
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "data_quality": validation_report['overall_quality'],
            "feature_engineering": {
                "total_features": engineered_features.get('feature_metadata', {}).get('total_features', 0),
                "feature_quality": engineered_features.get('feature_metadata', {}).get('feature_quality', 'unknown')
            },
            "analysis_results": analysis_results,
            "raw_data_summary": {
                "sources_collected": raw_data['metadata']['successful_sources'],
                "total_sources": raw_data['metadata']['total_sources']
            }
        }
        
        logger.info(f"✅ Market analysis completed: {final_report['analysis_id']}")
        
        return {
            "status": "success",
            "data": final_report
        }
        
    except Exception as e:
        logger.error(f"❌ Error in market analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@ml_analysis_router.get("/performance/metrics")
async def get_performance_metrics(model_name: Optional[str] = None, time_window: str = "24h"):
    """دریافت متریک‌های عملکرد مدل‌ها"""
    try:
        from ml_core import performance_tracker
        
        if model_name:
            # متریک‌های یک مدل خاص
            if model_name not in ml_model_manager.active_models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            performance_data = performance_tracker.get_model_performance(model_name, time_window)
        else:
            # متریک‌های همه مدل‌ها
            performance_data = performance_tracker.get_comparative_analysis()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": performance_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_analysis_router.get("/data/quality")
async def get_data_quality_report():
    """دریافت گزارش کیفیت داده‌ها"""
    try:
        # اعتبارسنجی داده‌های فعلی
        raw_data = await data_integrator.collect_raw_data()
        validation_report = data_validator.validate_data_quality(raw_data)
        
        # روندهای کیفیت
        quality_trends = data_validator.get_data_quality_trends()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "current_quality": validation_report,
                "quality_trends": quality_trends,
                "validation_history": data_validator.get_validation_history(24)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting data quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_analysis_router.get("/features/engineered")
async def get_engineered_features():
    """دریافت ویژگی‌های مهندسی شده آخر"""
    try:
        from debug_system.storage.cache_debugger import cache_debugger
        
        # دریافت از کش
        features = cache_debugger.get_data("utb", "engineered_features:latest")
        
        if not features:
            # اگر در کش نبود، جدید تولید کن
            raw_data = await data_integrator.collect_raw_data()
            features = feature_engineer.engineer_market_features(raw_data)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": features
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting engineered features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_analysis_router.get("/alerts/active")
async def get_active_alerts():
    """دریافت هشدارهای فعال سیستم"""
    try:
        from ml_core import performance_tracker
        
        alerts = performance_tracker.get_active_alerts()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_active_alerts": len(alerts),
                "alerts": alerts
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _prepare_model_input(engineered_features: Dict[str, Any]):
    """آماده‌سازی ورودی مدل از ویژگی‌های مهندسی شده"""
    # این تابع ویژگی‌ها را به تانسور تبدیل می‌کند
    # پیاده‌سازی ساده برای نمونه
    
    import torch
    import numpy as np
    
    try:
        # استخراج تمام مقادیر عددی از ویژگی‌ها
        numeric_values = []
        
        def extract_numbers(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        numeric_values.append(value)
                    elif isinstance(value, dict):
                        extract_numbers(value, f"{prefix}{key}.")
                    elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value[:5]):
                        numeric_values.extend(value[:5])
        
        extract_numbers(engineered_features)
        
        if numeric_values:
            # نرمال‌سازی و تبدیل به تانسور
            values_array = np.array(numeric_values)
            if len(values_array) > 100:  # محدود کردن تعداد ویژگی‌ها
                values_array = values_array[:100]
            elif len(values_array) < 100:  # padding اگر کم بود
                values_array = np.pad(values_array, (0, 100 - len(values_array)), 'constant')
            
            # تبدیل به شکل مورد نیاز مدل (batch_size, sequence_length, features)
            input_tensor = torch.FloatTensor(values_array).unsqueeze(0).unsqueeze(0)
            return input_tensor
        else:
            raise ValueError("No numeric features found for model input")
            
    except Exception as e:
        logger.error(f"❌ Error preparing model input: {e}")
        raise
