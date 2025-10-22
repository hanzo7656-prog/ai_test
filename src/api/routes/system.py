# 📁 src/api/routes/system.py

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
import psutil
import os
from datetime import datetime

from ...monitoring.health_check import SystemHealthChecker
from ...core.engine import CryptoAnalysisEngine
from ..middleware import verify_api_key

router = APIRouter()

@router.get("/health")
async def system_health(
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """بررسی سلامت کامل سیستم"""
    try:
        # وضعیت موتور تحلیل
        engine_status = engine.get_system_status()
        
        # وضعیت سیستم
        system_status = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': os.times().elapsed,
            'timestamp': datetime.now().isoformat()
        }
        
        # وضعیت کامپوننت‌ها
        components_status = {
            'data_manager': 'healthy' if engine.data_manager else 'failed',
            'signal_engine': 'healthy' if engine.signal_engine else 'failed',
            'transformer': 'healthy' if engine.spiking_transformer else 'failed',
            'ai_models': 'healthy' if engine.regime_classifier and engine.pattern_predictor else 'degraded'
        }
        
        return {
            'system': system_status,
            'engine': engine_status,
            'components': components_status,
            'overall_status': 'healthy' if all(
                status == 'healthy' for status in components_status.values()
            ) else 'degraded'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/performance")
async def system_performance(
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """معیارهای عملکرد سیستم"""
    try:
        performance_data = engine.performance_tracker.get_summary()
        memory_data = engine.memory_monitor.get_usage_stats()
        
        return {
            'performance_metrics': performance_data,
            'memory_usage': memory_data,
            'analysis_cycles': engine.health_checker.successful_cycles,
            'success_rate': engine.health_checker.successful_cycles / max(
                1, engine.health_checker.successful_cycles + engine.health_checker.failed_cycles
            ) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance check failed: {str(e)}")

@router.post("/maintenance/clear-cache")
async def clear_cache(
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """پاک‌سازی کش سیستم"""
    try:
        if hasattr(engine.data_manager, 'cache'):
            engine.data_manager.cache.clear()
        
        # پاک‌سازی حافظه
        import gc
        gc.collect()
        
        return {
            'message': 'Cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clearance failed: {str(e)}")

@router.get("/config")
async def get_configuration(
    engine: CryptoAnalysisEngine = Depends(get_analysis_engine),
    api_key: str = Depends(verify_api_key)
):
    """دریافت پیکربندی سیستم"""
    try:
        return {
            'system_config': engine.config,
            'feature_flags': {
                'spiking_transformer': engine.config.get('transformer', {}).get('enabled', True),
                'ai_models': engine.config.get('ai_models', {}).get('enabled', True),
                'multi_timeframe': engine.config.get('multi_timeframe', {}).get('enabled', True)
            },
            'symbols': engine.config.get('default_symbols', []),
            'analysis_interval': engine.config.get('analysis_interval', 300)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")
