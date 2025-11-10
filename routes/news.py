from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

news_router = APIRouter(prefix="/api/news", tags=["News"])

@news_router.get("/all", summary="اخبار عمومی")
async def get_news(
    limit: int = Query(50, ge=1, le=100, description="تعداد اخبار (۱ تا ۱۰۰)"),
    page: int = Query(1, ge=1, description="شماره صفحه")
):
    """دریافت اخبار عمومی از CoinStats API"""
    try:
        logger.info(f"📰 Fetching news - Limit: {limit}")
        
        raw_data = coin_stats_manager.get_news(limit=limit)
        
        if "error" in raw_data:
            logger.error(f"❌ News API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # بررسی ساختار داده‌ها
        if isinstance(raw_data, dict):
            news_items = raw_data.get('data', raw_data.get('result', []))
        elif isinstance(raw_data, list):
            news_items = raw_data
        else:
            news_items = []
        
        # پردازش ساده داده‌ها
        processed_news = []
        for news_item in news_items[:limit]:
            processed_news.append({
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'description': news_item.get('description'),
                'url': news_item.get('url'),
                'source': news_item.get('source'),
                'published_at': news_item.get('published_at', news_item.get('publishedAt')),
                'image_url': news_item.get('imageUrl'),
                'tags': news_item.get('tags', []),
                'categories': news_item.get('categories', []),
                'last_updated': datetime.now().isoformat()
            })
        
        response = {
            'status': 'success',
            'data': processed_news,
            'meta': {
                'total': len(processed_news),
                'limit': limit,
                'page': page
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ News fetched successfully - Total: {len(processed_news)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Unexpected error in news: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/type/{news_type}", summary="اخبار بر اساس نوع")
async def get_news_by_type(
    news_type: str,
    limit: int = Query(10, ge=1, le=50, description="تعداد اخبار (۱ تا ۵۰)")
):
    """دریافت اخبار بر اساس نوع"""
    try:
        # انواع معتبر
        valid_types = ["latest", "trending", "featured", "breaking", "analysis", "handpicked"]
        if news_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid news type. Valid types: {valid_types}")
        
        logger.info(f"📰 Fetching {news_type} news - Limit: {limit}")
        
        raw_data = coin_stats_manager.get_news_by_type(news_type, limit=limit)
        
        if "error" in raw_data:
            logger.error(f"❌ {news_type} news API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # بررسی ساختار داده‌ها
        if isinstance(raw_data, dict):
            news_items = raw_data.get('data', raw_data.get('result', []))
        elif isinstance(raw_data, list):
            news_items = raw_data
        else:
            news_items = []
        
        processed_news = []
        for news_item in news_items[:limit]:
            processed_news.append({
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'description': news_item.get('description'),
                'url': news_item.get('url'),
                'source': news_item.get('source'),
                'published_at': news_item.get('published_at', news_item.get('publishedAt')),
                'image_url': news_item.get('imageUrl'),
                'type': news_type,
                'tags': news_item.get('tags', []),
                'last_updated': datetime.now().isoformat()
            })
        
        response = {
            'status': 'success',
            'data': processed_news,
            'meta': {
                'type': news_type,
                'total': len(processed_news),
                'limit': limit
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ {news_type} news fetched successfully - Total: {len(processed_news)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error in {news_type} news: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/sources", summary="منابع خبری")
async def get_news_sources():
    """دریافت لیست منابع خبری"""
    try:
        logger.info("📰 Fetching news sources")
        
        raw_data = coin_stats_manager.get_news_sources()
        
        if "error" in raw_data:
            logger.error(f"❌ News sources API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # 🔥 رفع مشکل: بررسی مستقیم نوع داده
        if isinstance(raw_data, list):
            sources = raw_data
        else:
            sources = []
        
        processed_sources = []
        for source in sources:
            if isinstance(source, dict):
                processed_sources.append({
                    'id': source.get('id', 'unknown'),
                    'name': source.get('name', 'Unknown'),
                    'url': source.get('url', ''),
                    'description': source.get('description', ''),
                    'language': source.get('language', 'en'),
                    'country': source.get('country', ''),
                    'category': source.get('category', 'crypto'),
                    'last_updated': datetime.now().isoformat()
                })
            else:
                # اگر دیکشنری نیست، به صورت ساده ذخیره کن
                processed_sources.append({
                    'id': 'unknown',
                    'name': str(source),
                    'raw_data': source
                })
        
        response = {
            'status': 'success',
            'data': processed_sources,
            'meta': {
                'total': len(processed_sources)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ News sources fetched successfully - Total: {len(processed_sources)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error in news sources: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
@news_router.get("/detail/{news_id}", summary="جزئیات خبر")
async def get_news_detail(news_id: str):
    """دریافت جزئیات کامل یک خبر"""
    try:
        logger.info(f"📰 Fetching news detail: {news_id}")
        
        raw_data = coin_stats_manager.get_news_detail(news_id)
        
        if "error" in raw_data:
            logger.error(f"❌ News detail API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # بررسی ساختار داده‌ها
        if isinstance(raw_data, dict):
            news_data = raw_data.get('data', raw_data.get('result', {}))
        else:
            news_data = raw_data if raw_data else {}
        
        # اگر news_data دیکشنری نیست، از raw_data استفاده کن
        if not isinstance(news_data, dict):
            news_data = raw_data if isinstance(raw_data, dict) else {}
        
        processed_detail = {
            'id': news_data.get('id', news_id),
            'title': news_data.get('title'),
            'content': news_data.get('content', news_data.get('description')),
            'url': news_data.get('url'),
            'source': news_data.get('source'),
            'author': news_data.get('author'),
            'published_at': news_data.get('published_at', news_data.get('publishedAt')),
            'image_url': news_data.get('imageUrl'),
            'tags': news_data.get('tags', []),
            'categories': news_data.get('categories', []),
            'last_updated': datetime.now().isoformat(),
            'raw_data_available': bool(news_data)
        }
        
        response = {
            'status': 'success',
            'data': processed_detail,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ News detail fetched successfully: {news_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error in news detail {news_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/debug/{news_type}", summary="دیباگ اخبار")
async def debug_news_data(news_type: str = "handpicked"):
    """ابزار دیباگ برای بررسی ساختار داده اخبار"""
    try:
        raw_data = coin_stats_manager.get_news_by_type(news_type)
        
        # نمونه‌ای از داده‌ها برای نمایش
        if isinstance(raw_data, dict):
            sample_data = raw_data.get('data', raw_data.get('result', []))
        elif isinstance(raw_data, list):
            sample_data = raw_data
        else:
            sample_data = []
            
        sample_item = sample_data[0] if sample_data else {}
        
        return {
            'status': 'debug',
            'endpoint': f"news/type/{news_type}",
            'raw_data_type': type(raw_data).__name__,
            'manager_response_keys': list(raw_data.keys()) if isinstance(raw_data, dict) else 'not_dict',
            'data_count': len(sample_data),
            'sample_item_structure': {
                'keys': list(sample_item.keys()) if sample_item else 'no_data',
                'sample_values': {k: type(v).__name__ for k, v in list(sample_item.items())[:5]} if sample_item else 'no_data'
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
