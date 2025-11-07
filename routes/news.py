from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

news_router = APIRouter(prefix="/api/news", tags=["News"])

@news_router.get("/all", summary="اخبار عمومی")
async def get_news(limit: int = Query(50, ge=1, le=100)):
    """دریافت اخبار پردازش شده عمومی"""
    try:
        raw_data = coin_stats_manager.get_news()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # محدود کردن نتایج بر اساس limit
        news_items = raw_data.get('result', [])[:limit]
        
        processed_news = []
        for news_item in news_items:
            processed_news.append({
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'description': news_item.get('description'),
                'url': news_item.get('url'),
                'source': news_item.get('source'),
                'published_at': news_item.get('publishedAt'),
                'sentiment': _analyze_sentiment(news_item),
                'importance': _calculate_importance(news_item),
                'tags': news_item.get('tags', []),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_news,
            'total': len(processed_news),
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@news_router.get("/type/{news_type}", summary="اخبار بر اساس نوع")
async def get_news_by_type(
    news_type: str,
    limit: int = Query(10, ge=1, le=50)
):
    """دریافت اخبار پردازش شده بر اساس نوع"""
    try:
        raw_data = coin_stats_manager.get_news_by_type(news_type)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # محدود کردن نتایج
        news_items = raw_data.get('result', [])[:limit]
        
        processed_news = []
        for news_item in news_items:
            processed_news.append({
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'description': news_item.get('description'),
                'url': news_item.get('url'),
                'source': news_item.get('source'),
                'published_at': news_item.get('publishedAt'),
                'type': news_type,
                'sentiment': _analyze_sentiment(news_item),
                'importance': _calculate_importance(news_item),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_news,
            'type': news_type,
            'total': len(processed_news),
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in news by type {news_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@news_router.get("/sources", summary="منابع خبری")
async def get_news_sources():
    """دریافت منابع خبری پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_news_sources()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        processed_sources = []
        for source in raw_data.get('result', []):
            processed_sources.append({
                'id': source.get('id'),
                'name': source.get('name'),
                'url': source.get('url'),
                'reliability_score': _calculate_reliability(source),
                'coverage': source.get('coverage', 'general'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_sources,
            'total': len(processed_sources),
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in news sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@news_router.get("/detail/{news_id}", summary="جزئیات خبر")
async def get_news_detail(news_id: str):
    """دریافت جزئیات پردازش شده یک خبر"""
    try:
        raw_data = coin_stats_manager.get_news_detail(news_id)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        processed_detail = {
            'id': raw_data.get('id'),
            'title': raw_data.get('title'),
            'content': raw_data.get('content'),
            'url': raw_data.get('url'),
            'source': raw_data.get('source'),
            'author': raw_data.get('author'),
            'published_at': raw_data.get('publishedAt'),
            'sentiment': _analyze_sentiment(raw_data),
            'importance': _calculate_importance(raw_data),
            'summary': _generate_summary(raw_data),
            'key_points': _extract_key_points(raw_data),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_detail,
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in news detail {news_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# توابع کمکی پردازش اخبار
def _analyze_sentiment(news_item: Dict) -> str:
    """تحلیل احساسات خبر"""
    title = news_item.get('title', '').lower()
    description = news_item.get('description', '').lower()
    
    positive_words = ['bullish', 'surge', 'rally', 'gain', 'positive', 'growth']
    negative_words = ['bearish', 'drop', 'crash', 'loss', 'negative', 'decline']
    
    content = title + ' ' + description
    
    positive_count = sum(1 for word in positive_words if word in content)
    negative_count = sum(1 for word in negative_words if word in content)
    
    if positive_count > negative_count:
        return "bullish"
    elif negative_count > positive_count:
        return "bearish"
    else:
        return "neutral"

def _calculate_importance(news_item: Dict) -> int:
    """محاسبه اهمیت خبر"""
    score = 0
    
    # منبع معتبر
    reliable_sources = ['cointelegraph', 'decrypt', 'coindesk']
    source = news_item.get('source', '').lower()
    if any(rel_source in source for rel_source in reliable_sources):
        score += 3
    
    # طول محتوا
    content_length = len(news_item.get('description', ''))
    if content_length > 500:
        score += 2
    elif content_length > 200:
        score += 1
    
    # تگ‌های مهم
    important_tags = ['bitcoin', 'ethereum', 'regulation', 'adoption']
    tags = news_item.get('tags', [])
    if any(tag in important_tags for tag in tags):
        score += 2
    
    return min(score, 5)  # نمره ۱ تا ۵

def _calculate_reliability(source: Dict) -> int:
    """محاسبه قابلیت اطمینان منبع"""
    reliable_sources = {
        'cointelegraph': 5,
        'decrypt': 4, 
        'coindesk': 4,
        'newsbtc': 3,
        'cryptopotato': 3
    }
    
    source_name = source.get('name', '').lower()
    for rel_source, score in reliable_sources.items():
        if rel_source in source_name:
            return score
    
    return 2  # پیش‌فرض

def _generate_summary(news_item: Dict) -> str:
    """تولید خلاصه خبر"""
    description = news_item.get('description', '')
    if len(description) > 150:
        return description[:147] + '...'
    return description

def _extract_key_points(news_item: Dict) -> List[str]:
    """استخراج نکات کلیدی"""
    # در یک پیاده‌سازی واقعی از NLP استفاده می‌شود
    content = news_item.get('description', '')
    sentences = content.split('.')
    
    key_points = []
    for sentence in sentences[:3]:  # فقط ۳ جمله اول
        sentence = sentence.strip()
        if len(sentence) > 20:  # جملات کوتاه را نادیده بگیر
            key_points.append(sentence)
    
    return key_points
