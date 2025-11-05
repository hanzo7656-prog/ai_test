from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()

class AIAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"

@router.post("/analyze")
async def ai_analyze(request: AIAnalysisRequest):
    """آنالیز نماد توسط هوش مصنوعی"""
    try:
        from trading_ai.main_trading_system import main_trading_system
        
        if not main_trading_system.is_initialized:
            success = main_trading_system.initialize_system()
            if not success:
                raise HTTPException(status_code=500, detail="سیستم AI راه‌اندازی نشد")
            
        start_time = time.time()
        result = main_trading_system.analyze_symbol(
            request.symbol, 
            request.analysis_type
        )
        processing_time = time.time() - start_time
        
        logger.info(f"تحلیل AI برای {request.symbol} در {processing_time:.2f} ثانیه تکمیل شد")
        
        return {
            "status": "success",
            "data": result,
            "processing_time": processing_time,
            "timestamp": result.get('timestamp'),
            "ai_processed": True
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل AI برای {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def ai_status():
    """وضعیت سیستم AI"""
    try:
        from trading_ai.main_trading_system import main_trading_system
        status = main_trading_system.get_system_status()
        return {
            "status": "success",
            "ai_system": status,
            "timestamp": status.get('last_analysis_time')
        }
    except Exception as e:
        logger.error(f"خطا در دریافت وضعیت AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def ai_initialize():
    """راه‌اندازی سیستم AI"""
    try:
        from trading_ai.main_trading_system import main_trading_system
        success = main_trading_system.initialize_system()
        return {
            "status": "success" if success else "error",
            "initialized": success,
            "message": "سیستم AI راه‌اندازی شد" if success else "خطا در راه‌اندازی AI"
        }
    except Exception as e:
        logger.error(f"خطا در راه‌اندازی AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))
