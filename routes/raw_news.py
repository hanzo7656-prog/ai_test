from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

raw_news_router = APIRouter(prefix="/api/raw/news", tags=["Raw News"])

@raw_news_router.get("/all", summary="اخبار عمومی خام")
async def get_raw_news(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار عمومی خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_news(limit)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'limit': limit,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/type/{news_type}", summary="اخبار خام بر اساس نوع")
async def get_raw_news_by_type(
    news_type: str,
    limit: int = Query(10, ge=1, le=50)
):
    """دریافت اخبار خام بر اساس نوع - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_news_by_type(news_type, limit)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'news_type': news_type,
            'data': raw_data,
            'limit': limit,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw news by type {news_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/sources", summary="منابع خبری خام")
async def get_raw_news_sources():
    """دریافت منابع خبری خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_news_sources()
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw news sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/detail/{news_id}", summary="جزئیات خبر خام")
async def get_raw_news_detail(news_id: str):
    """دریافت جزئیات خبر خام - بدون پردازش"""
    try:
        raw_data = coin_stats_manager.get_news_detail(news_id)
        
        return {
            'status': 'success',
            'data_type': 'raw',
            'source': 'coinstats_api',
            'news_id': news_id,
            'data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in raw news detail {news_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
