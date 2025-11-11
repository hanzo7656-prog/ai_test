from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

try:
    from smart_cache_system import coins_cache, raw_coins_cache
    SMART_CACHE_AVAILABLE = True
except ImportError:
    from debug_system.storage.cache_decorators import cache_coins, cache_raw_coins
    SMART_CACHE_AVAILABLE = False

raw_news_router = APIRouter(prefix="/api/raw/news", tags=["Raw News"])

@raw_news_router.get("/all", summary="داده‌های خام اخبار عمومی")
@raw_news_cache
async def get_raw_news(limit: int = Query(50, ge=1, le=100)):
    """دریافت داده‌های خام اخبار عمومی از CoinStats API"""
    try:
        raw_data = coin_stats_manager.get_news()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌ها
        news_items = raw_data.get('data', [])
        limited_data = news_items[:limit]
        
        return {
            'status': 'success',
            'data_type': 'raw_news_feed',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'limit_applied': limit,
            'total_available': len(news_items),
            'data': {
                'result': limited_data,
                'meta': raw_data.get('meta', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/type/{news_type}", summary="داده‌های خام اخبار دسته‌بندی شده")
@raw_news_cache
async def get_raw_news_by_type(
    news_type: str,
    limit: int = Query(10, ge=1, le=50)
):
    """دریافت داده‌های خام اخبار بر اساس دسته‌بندی"""
    try:
        # انواع معتبر بر اساس مستندات
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid news type. Valid types: {valid_types}")
        
        raw_data = coin_stats_manager.get_news_by_type(news_type, limit=limit)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌ها
        news_items = raw_data.get('data', [])
        limited_data = news_items[:limit]
        
        return {
            'status': 'success',
            'data_type': 'raw_categorized_news',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'news_type': news_type,
            'timestamp': datetime.now().isoformat(),
            'limit_applied': limit,
            'total_available': len(news_items),
            'data': {
                'result': limited_data,
                'meta': raw_data.get('meta', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw news by type {news_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/sources", summary="داده‌های خام منابع خبری")
@raw_news_cache
async def get_raw_news_sources():
    """دریافت داده‌های خام منابع خبری از CoinStats API"""
    try:
        raw_data = coin_stats_manager.get_news_sources()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # داده‌ها مستقیماً از manager می‌آیند
        sources_list = raw_data.get('data', [])
        
        return {
            'status': 'success',
            'data_type': 'raw_news_sources',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'sources': sources_list,
                'total_count': len(sources_list)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw news sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/detail/{news_id}", summary="داده‌های خام جزئیات خبر")
@raw_news_cache
async def get_raw_news_detail(news_id: str):
    """دریافت داده‌های خام جزئیات کامل یک خبر"""
    try:
        raw_data = coin_stats_manager.get_news_detail(news_id)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data_type': 'raw_news_detail',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'news_id': news_id,
            'timestamp': datetime.now().isoformat(),
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw news detail {news_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/sentiment-analysis", summary="تحلیل احساسات اخبار")
@raw_news_cache
async def get_news_sentiment_analysis(
    limit: int = Query(20, ge=1, le=50),
    news_type: str = Query(None, description="نوع خبر: handpicked, trending, latest, bullish, bearish")
):
    """تحلیل احساسات اخبار از داده‌های واقعی"""
    try:
        if news_type:
            raw_data = coin_stats_manager.get_news_by_type(news_type)
        else:
            raw_data = coin_stats_manager.get_news()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        news_items = raw_data.get('data', [])[:limit]
        
        # تحلیل احساسات ساده
        sentiment_analysis = _perform_sentiment_analysis(news_items)
        
        return {
            'status': 'success',
            'data_type': 'news_sentiment_analysis',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'limit': limit,
                'news_type': news_type or 'all'
            },
            'sentiment_analysis': sentiment_analysis,
            'sample_data': news_items[:5]
        }
        
    except Exception as e:
        logger.error(f"Error in news sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/metadata", summary="متادیتای اخبار و منابع")
@raw_news_cache
async def get_news_metadata():
    """دریافت متادیتای کامل اخبار و منابع"""
    try:
        return {
            'status': 'success',
            'data_type': 'news_metadata',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'available_endpoints': [
                {
                    'endpoint': '/api/raw/news/all',
                    'description': 'داده‌های خام اخبار عمومی',
                    'parameters': ['limit'],
                    'use_case': 'تحلیل کلی اخبار بازار'
                },
                {
                    'endpoint': '/api/raw/news/type/{news_type}',
                    'description': 'داده‌های خام اخبار دسته‌بندی شده',
                    'parameters': ['news_type', 'limit'],
                    'news_types': ['handpicked', 'trending', 'latest', 'bullish', 'bearish']
                },
                {
                    'endpoint': '/api/raw/news/sources',
                    'description': 'داده‌های خام منابع خبری',
                    'parameters': [],
                    'use_case': 'تحلیل اعتبار و توزیع منابع'
                },
                {
                    'endpoint': '/api/raw/news/detail/{news_id}',
                    'description': 'داده‌های خام جزئیات کامل خبر',
                    'parameters': ['news_id'],
                    'use_case': 'تحلیل عمقی محتوای خبر'
                },
                {
                    'endpoint': '/api/raw/news/sentiment-analysis',
                    'description': 'تحلیل احساسات اخبار',
                    'parameters': ['limit', 'news_type'],
                    'use_case': 'آموزش مدل‌های تحلیل احساسات'
                }
            ],
            'news_categories': {
                'handpicked': 'اخبار منتخب و مهم',
                'trending': 'اخبار داغ و پرطرفدار',
                'latest': 'آخرین اخبار',
                'bullish': 'اخبار مثبت و صعودی',
                'bearish': 'اخبار منفی و نزولی'
            },
            'data_structure': {
                'news_item': {
                    'id': 'شناسه یکتای خبر',
                    'title': 'عنوان خبر',
                    'description': 'خلاصه خبر',
                    'url': 'لینک منبع اصلی',
                    'source': 'منبع خبر',
                    'publishedAt': 'زمان انتشار',
                    'tags': 'تگ‌های موضوعی',
                    'content': 'محتوای کامل (در جزئیات خبر)'
                },
                'news_source': {
                    'sourcename': 'نام منبع',
                    'weburl': 'آدرس وبسایت',
                    'feedurl': 'آدرس فید RSS',
                    'sourceImg': 'آدرس لوگو'
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in news metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/debug/simple", summary="دیباگ ساده مدیر")
@raw_news_cache
async def debug_simple():
    """دیباگ ساده برای بررسی manager"""
    try:
        # تست مستقیم بدون پردازش
        test_data = coin_stats_manager.get_news_sources()
        
        return {
            'status': 'debug',
            'timestamp': datetime.now().isoformat(),
            'data_type': type(test_data).__name__,
            'is_list': isinstance(test_data, list),
            'is_dict': isinstance(test_data, dict),
            'length': len(test_data) if hasattr(test_data, '__len__') else 'no_length',
            'raw_preview': str(test_data)[:500] if test_data else 'empty'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'timestamp': datetime.now().isoformat()
        }

# ============================ توابع کمکی ============================

def _perform_sentiment_analysis(news_items: List[Dict]) -> Dict[str, Any]:
    """انجام تحلیل احساسات ساده روی اخبار"""
    if not news_items:
        return {'analysis': 'no_news_data_available'}
    
    sentiment_results = []
    
    for news in news_items:
        sentiment = _analyze_basic_sentiment(news)
        sentiment_results.append({
            'news_id': news.get('id'),
            'title': news.get('title'),
            'sentiment': sentiment
        })
    
    # آمار کلی احساسات
    sentiment_counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for result in sentiment_results:
        sentiment_counts[result['sentiment']] += 1
    
    return {
        'total_analyzed': len(news_items),
        'sentiment_distribution': sentiment_counts,
        'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get),
        'detailed_results': sentiment_results[:5]
    }

def _analyze_basic_sentiment(news: Dict) -> str:
    """تحلیل احساسات پایه بر اساس کلمات کلیدی"""
    text = (news.get('title', '') + ' ' + news.get('description', '')).lower()
    
    positive_words = ['bullish', 'surge', 'rally', 'gain', 'positive', 'growth', 'up', 'rise', 'profit']
    negative_words = ['bearish', 'drop', 'crash', 'loss', 'negative', 'decline', 'down', 'fall', 'warning']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'
