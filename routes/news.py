from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

try:
    from debug_system.storage.cache_decorators import cache_coins_with_archive as coins_cache
    logger.info("✅ Cache System: Archive Enabled")
except ImportError as e:
    logger.error(f"❌ Cache system unavailable: {e}")
    # Fallback نهایی
    def cache_news_with_archive(func):
        return func

news_router = APIRouter(prefix="/api/news", tags=["News"])

@news_router.get("/all", summary="اخبار عمومی")
@cache_news_with_archive()
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
        
        # پردازش داده‌ها
        news_items = raw_data.get('data', [])
        
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
@cache_news_with_archive()
async def get_news_by_type(
    news_type: str,
    limit: int = Query(10, ge=1, le=50, description="تعداد اخبار (۱ تا ۵۰)")
):
    """دریافت اخبار بر اساس نوع"""
    try:
        # 🔥 انواع معتبر بر اساس مستندات
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid news type. Valid types: {valid_types}")
        
        logger.info(f"📰 Fetching {news_type} news - Limit: {limit}")
        
        raw_data = coin_stats_manager.get_news_by_type(news_type, limit=limit)
        
        if "error" in raw_data:
            logger.error(f"❌ {news_type} news API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌ها
        news_items = raw_data.get('data', [])
        
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
@cache_news_with_archive()
async def get_news_sources():
    """دریافت لیست منابع خبری"""
    try:
        logger.info("📰 Fetching news sources")
        
        raw_data = coin_stats_manager.get_news_sources()
        
        if "error" in raw_data:
            logger.error(f"❌ News sources API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌های منابع خبری
        sources_list = raw_data.get('data', [])
        
        processed_sources = []
        for source in sources_list:
            if isinstance(source, dict):
                processed_sources.append({
                    'id': source.get('sourcename', '').lower().replace(' ', '-'),
                    'name': source.get('sourcename', 'Unknown'),
                    'url': source.get('weburl', ''),
                    'logo': source.get('sourceImg', source.get('logo', '')),
                    'feed_url': source.get('feedurl', ''),
                    'description': f"News source: {source.get('sourcename', 'Unknown')}",
                    'language': 'en',
                    'category': 'crypto',
                    'last_updated': datetime.now().isoformat()
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
@cache_news_with_archive()
async def get_news_detail(news_id: str):
    """دریافت جزئیات کامل یک خبر"""
    try:
        logger.info(f"📰 Fetching news detail: {news_id}")
        
        raw_data = coin_stats_manager.get_news_detail(news_id)
        
        if "error" in raw_data:
            logger.error(f"❌ News detail API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌های جزئیات خبر
        news_data = raw_data.get('data', {})
        
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
            'last_updated': datetime.now().isoformat()
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

@news_router.get("/categories", summary="دسته‌بندی‌های خبر")
@cache_news_with_archive()
async def get_news_categories():
    """دریافت دسته‌بندی‌های خبری موجود"""
    try:
        categories = [
            {
                'id': 'handpicked',
                'name': 'اخبار منتخب',
                'description': 'اخبار مهم و منتخب شده توسط تیم تحریریه',
                'icon': '⭐'
            },
            {
                'id': 'trending',
                'name': 'اخبار داغ',
                'description': 'اخبار پرطرفدار و ترند شده',
                'icon': '🔥'
            },
            {
                'id': 'latest',
                'name': 'آخرین اخبار',
                'description': 'جدیدترین اخبار بازار کریپتو',
                'icon': '🆕'
            },
            {
                'id': 'bullish',
                'name': 'اخبار مثبت',
                'description': 'اخبار با احساسات مثبت و صعودی',
                'icon': '📈'
            },
            {
                'id': 'bearish',
                'name': 'اخبار منفی',
                'description': 'اخبار با احساسات منفی و نزولی',
                'icon': '📉'
            }
        ]
        
        response = {
            'status': 'success',
            'data': categories,
            'meta': {
                'total': len(categories)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ News categories fetched successfully")
        return response
        
    except Exception as e:
        logger.error(f"🚨 Error in news categories: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/search", summary="جستجو در اخبار")
@cache_news_with_archive()
async def search_news(
    query: str = Query(..., description="عبارت جستجو"),
    limit: int = Query(20, ge=1, le=50, description="تعداد نتایج")
):
    """جستجو در اخبار بر اساس کلمات کلیدی"""
    try:
        logger.info(f"🔍 Searching news - Query: {query}, Limit: {limit}")
        
        # دریافت اخبار عمومی
        raw_data = coin_stats_manager.get_news(limit=100)
        
        if "error" in raw_data:
            logger.error(f"❌ News search API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # فیلتر کردن نتایج بر اساس جستجو
        news_items = raw_data.get('data', [])
        query_lower = query.lower()
        
        filtered_news = []
        for news_item in news_items:
            title = news_item.get('title', '').lower()
            description = news_item.get('description', '').lower()
            
            if query_lower in title or query_lower in description:
                filtered_news.append({
                    'id': news_item.get('id'),
                    'title': news_item.get('title'),
                    'description': news_item.get('description'),
                    'url': news_item.get('url'),
                    'source': news_item.get('source'),
                    'published_at': news_item.get('published_at', news_item.get('publishedAt')),
                    'image_url': news_item.get('imageUrl'),
                    'tags': news_item.get('tags', []),
                    'last_updated': datetime.now().isoformat()
                })
        
        # اعمال محدودیت
        limited_results = filtered_news[:limit]
        
        response = {
            'status': 'success',
            'data': limited_results,
            'meta': {
                'query': query,
                'total_found': len(filtered_news),
                'returned': len(limited_results),
                'limit': limit
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ News search completed - Found: {len(filtered_news)}, Returned: {len(limited_results)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error in news search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/stats", summary="آمار اخبار")
@cache_news_with_archive()
async def get_news_stats():
    """دریافت آمار و تحلیل کلی اخبار"""
    try:
        logger.info("📊 Fetching news statistics")
        
        # دریافت اخبار از انواع مختلف
        latest_news = coin_stats_manager.get_news_by_type("latest", limit=100)
        trending_news = coin_stats_manager.get_news_by_type("trending", limit=100)
        handpicked_news = coin_stats_manager.get_news_by_type("handpicked", limit=100)
        
        # تحلیل داده‌ها
        total_news = len(latest_news.get('data', []))
        
        # تحلیل منابع
        sources_data = coin_stats_manager.get_news_sources()
        total_sources = len(sources_data.get('data', []))
        
        # تحلیل احساسات
        sentiment_analysis = _analyze_news_sentiment(latest_news.get('data', []))
        
        response = {
            'status': 'success',
            'data': {
                'total_news': total_news,
                'total_sources': total_sources,
                'categories_count': 5,  # handpicked, trending, latest, bullish, bearish
                'sentiment_analysis': sentiment_analysis,
                'last_updated': datetime.now().isoformat()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ News statistics fetched successfully")
        return response
        
    except Exception as e:
        logger.error(f"🚨 Error in news stats: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/debug/{news_type}", summary="دیباگ اخبار")
@cache_news_with_archive()
async def debug_news_data(news_type: str = "handpicked"):
    """ابزار دیباگ برای بررسی ساختار داده اخبار"""
    try:
        raw_data = coin_stats_manager.get_news_by_type(news_type)
        
        # نمونه‌ای از داده‌ها برای نمایش
        news_items = raw_data.get('data', [])
        sample_item = news_items[0] if news_items else {}
        
        return {
            'status': 'debug',
            'endpoint': f"news/type/{news_type}",
            'raw_data_type': type(raw_data).__name__,
            'data_count': len(news_items),
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

# ============================ توابع کمکی ============================

def _analyze_news_sentiment(news_items: List[Dict]) -> Dict[str, Any]:
    """تحلیل احساسات اخبار"""
    if not news_items:
        return {'analysis': 'no_news_data_available'}
    
    sentiment_counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for news in news_items:
        sentiment = _analyze_basic_sentiment(news)
        sentiment_counts[sentiment] += 1
    
    total = len(news_items)
    
    return {
        'total_analyzed': total,
        'sentiment_distribution': sentiment_counts,
        'percentages': {
            'positive': round((sentiment_counts['positive'] / total) * 100, 2) if total > 0 else 0,
            'negative': round((sentiment_counts['negative'] / total) * 100, 2) if total > 0 else 0,
            'neutral': round((sentiment_counts['neutral'] / total) * 100, 2) if total > 0 else 0
        },
        'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get) if total > 0 else 'neutral'
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
