from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

raw_news_router = APIRouter(prefix="/api/raw/news", tags=["Raw News"])

@raw_news_router.get("/all", summary="داده‌های خام اخبار عمومی")
async def get_raw_news(limit: int = Query(50, ge=1, le=100)):
    """دریافت داده‌های خام اخبار عمومی از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_news()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # محدود کردن نتایج در سمت سرور
        news_items = raw_data.get('result', [])[:limit]
        
        # تحلیل داده‌های خبری
        news_analysis = _analyze_news_data(news_items)
        
        return {
            'status': 'success',
            'data_type': 'raw_news_feed',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'limit_applied': limit,
            'total_available': len(raw_data.get('result', [])),
            'analysis': news_analysis,
            'data': {
                'result': news_items,
                'meta': raw_data.get('meta', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/type/{news_type}", summary="داده‌های خام اخبار دسته‌بندی شده")
async def get_raw_news_by_type(
    news_type: str,
    limit: int = Query(10, ge=1, le=50)
):
    """دریافت داده‌های خام اخبار بر اساس دسته‌بندی از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_news_by_type(news_type)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # محدود کردن نتایج
        news_items = raw_data.get('result', [])[:limit]
        
        # تحلیل اخبار دسته‌بندی شده
        category_analysis = _analyze_news_by_category(news_items, news_type)
        
        return {
            'status': 'success',
            'data_type': 'raw_categorized_news',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'news_type': news_type,
            'timestamp': datetime.now().isoformat(),
            'limit_applied': limit,
            'total_available': len(raw_data.get('result', [])),
            'category_analysis': category_analysis,
            'data': {
                'result': news_items,
                'meta': raw_data.get('meta', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw news by type {news_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/sources", summary="داده‌های خام منابع خبری")
async def get_raw_news_sources():
    """دریافت داده‌های خام منابع خبری از CoinStats API"""
    try:
        raw_data = coin_stats_manager.get_news_sources()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # 🔥 رفع مشکل: بررسی نوع داده
        if isinstance(raw_data, list):
            sources_list = raw_data
        else:
            sources_list = []
        
        # تحلیل منابع خبری
        sources_analysis = _analyze_news_sources(sources_list)
        
        return {
            'status': 'success',
            'data_type': 'raw_news_sources',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'timestamp': datetime.now().isoformat(),
            'analysis': sources_analysis,
            'data': {
                'sources': sources_list,
                'raw_response': raw_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error in raw news sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@raw_news_router.get("/detail/{news_id}", summary="داده‌های خام جزئیات خبر")
async def get_raw_news_detail(news_id: str):
    """دریافت داده‌های خام جزئیات کامل یک خبر از CoinStats API - داده‌های واقعی برای هوش مصنوعی"""
    try:
        raw_data = coin_stats_manager.get_news_detail(news_id)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # تحلیل محتوای خبر
        content_analysis = _analyze_news_content(raw_data)
        
        return {
            'status': 'success',
            'data_type': 'raw_news_detail',
            'source': 'coinstats_api',
            'api_version': 'v1',
            'news_id': news_id,
            'timestamp': datetime.now().isoformat(),
            'content_analysis': content_analysis,
            'data': raw_data
        }
        
    except Exception as e:
        logger.error(f"Error in raw news detail {news_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/sentiment-analysis", summary="تحلیل احساسات اخبار")
async def get_news_sentiment_analysis(
    limit: int = Query(20, ge=1, le=50),
    news_type: str = Query(None, description="نوع خبر: handpicked, trending, latest, bullish, bearish")
):
    """تحلیل احساسات اخبار از داده‌های واقعی - برای آموزش هوش مصنوعی"""
    try:
        if news_type:
            raw_data = coin_stats_manager.get_news_by_type(news_type)
        else:
            raw_data = coin_stats_manager.get_news()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        news_items = raw_data.get('result', [])[:limit]
        
        # تحلیل احساسات پیشرفته
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
            'sample_data': news_items[:5]  # نمونه‌ای از داده‌ها برای آموزش
        }
        
    except Exception as e:
        logger.error(f"Error in news sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/metadata", summary="متادیتای اخبار و منابع")
async def get_news_metadata():
    """دریافت متادیتای کامل اخبار و منابع - برای آموزش هوش مصنوعی"""
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
                    'id': 'شناسه منبع',
                    'name': 'نام منبع',
                    'url': 'آدرس وبسایت',
                    'coverage': 'پوشش موضوعی'
                }
            },
            'field_descriptions': _get_news_field_descriptions()
        }
        
    except Exception as e:
        logger.error(f"Error in news metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@raw_news_router.get("/debug/manager", summary="دیباگ مدیر اخبار")
async def debug_news_manager():
    """بررسی ساختار داده‌های برگشتی از coin_stats_manager"""
    try:
        # تست تمام متدها
        news_data = coin_stats_manager.get_news()
        news_by_type = coin_stats_manager.get_news_by_type("latest")
        sources_data = coin_stats_manager.get_news_sources()
        
        return {
            'status': 'debug',
            'timestamp': datetime.now().isoformat(),
            'get_news_structure': {
                'type': type(news_data).__name__,
                'keys': list(news_data.keys()) if isinstance(news_data, dict) else 'not_dict',
                'sample': str(news_data)[:200] if news_data else 'empty'
            },
            'get_news_by_type_structure': {
                'type': type(news_by_type).__name__,
                'keys': list(news_by_type.keys()) if isinstance(news_by_type, dict) else 'not_dict',
                'sample': str(news_by_type)[:200] if news_by_type else 'empty'
            },
            'get_news_sources_structure': {
                'type': type(sources_data).__name__,
                'is_list': isinstance(sources_data, list),
                'length': len(sources_data) if isinstance(sources_data, list) else 'not_list',
                'sample': sources_data[:3] if isinstance(sources_data, list) and sources_data else 'empty'
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
# ============================ توابع کمکی برای هوش مصنوعی ============================

def _analyze_news_data(news_items: List[Dict]) -> Dict[str, Any]:
    """تحلیل داده‌های خبری"""
    if not news_items:
        return {'analysis': 'no_news_data_available'}
    
    # تحلیل منابع
    sources = {}
    tags = {}
    sentiment_distribution = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for news in news_items:
        # تحلیل منابع
        source = news.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
        
        # تحلیل تگ‌ها
        news_tags = news.get('tags', [])
        for tag in news_tags:
            tags[tag] = tags.get(tag, 0) + 1
        
        # تحلیل احساسات اولیه
        sentiment = _analyze_basic_sentiment(news)
        sentiment_distribution[sentiment] += 1
    
    # تحلیل زمانی
    timestamps = [news.get('publishedAt') for news in news_items if news.get('publishedAt')]
    recent_news = len([ts for ts in timestamps if _is_recent(ts)])
    
    return {
        'total_news': len(news_items),
        'source_distribution': {
            'total_sources': len(sources),
            'top_sources': dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5])
        },
        'tag_analysis': {
            'total_tags': len(tags),
            'popular_tags': dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10])
        },
        'sentiment_distribution': sentiment_distribution,
        'temporal_analysis': {
            'recent_news': recent_news,
            'coverage_period': f"{len(timestamps)} items with timestamps"
        },
        'content_analysis': {
            'average_title_length': sum(len(news.get('title', '')) for news in news_items) / len(news_items),
            'has_descriptions': len([news for news in news_items if news.get('description')]),
            'has_urls': len([news for news in news_items if news.get('url')])
        }
    }

def _analyze_news_by_category(news_items: List[Dict], category: str) -> Dict[str, Any]:
    """تحلیل اخبار دسته‌بندی شده"""
    base_analysis = _analyze_news_data(news_items)
    
    # تحلیل مختص دسته‌بندی
    category_specific = {
        'category': category,
        'category_characteristics': _get_category_characteristics(category),
        'expected_sentiment': _get_expected_sentiment_for_category(category)
    }
    
    return {**base_analysis, **category_specific}

def _analyze_news_sources(sources: List[Dict]) -> Dict[str, Any]:
    """تحلیل منابع خبری"""
    if not sources:
        return {'analysis': 'no_sources_data_available'}
    
    # تحلیل پوشش موضوعی
    coverage_types = {}
    for source in sources:
        coverage = source.get('coverage', 'general')
        coverage_types[coverage] = coverage_types.get(coverage, 0) + 1
    
    return {
        'total_sources': len(sources),
        'coverage_distribution': coverage_types,
        'source_reliability_metrics': {
            'has_urls': len([s for s in sources if s.get('url')]),
            'has_names': len([s for s in sources if s.get('name')]),
            'unique_sources': len(set(s.get('id') for s in sources if s.get('id')))
        }
    }

def _analyze_news_content(news_detail: Dict) -> Dict[str, Any]:
    """تحلیل محتوای خبر"""
    if not news_detail:
        return {'analysis': 'no_content_available'}
    
    title = news_detail.get('title', '')
    description = news_detail.get('description', '')
    content = news_detail.get('content', '')
    
    # تحلیل محتوای متنی
    text_analysis = {
        'title_length': len(title),
        'description_length': len(description),
        'content_length': len(content),
        'total_text_length': len(title) + len(description) + len(content),
        'has_content': bool(content.strip()),
        'has_description': bool(description.strip())
    }
    
    # تحلیل احساسات پیشرفته‌تر
    advanced_sentiment = _analyze_advanced_sentiment(title + ' ' + description + ' ' + content)
    
    # تحلیل کلمات کلیدی
    keywords = _extract_keywords(title + ' ' + description)
    
    return {
        'text_analysis': text_analysis,
        'sentiment_analysis': advanced_sentiment,
        'keyword_analysis': {
            'extracted_keywords': keywords[:10],
            'total_keywords': len(keywords)
        },
        'content_quality': {
            'score': _rate_content_quality(news_detail),
            'factors': _get_content_quality_factors(news_detail)
        }
    }

def _perform_sentiment_analysis(news_items: List[Dict]) -> Dict[str, Any]:
    """انجام تحلیل احساسات پیشرفته روی اخبار"""
    sentiment_results = []
    
    for news in news_items:
        sentiment = _analyze_advanced_sentiment(
            news.get('title', '') + ' ' + news.get('description', '')
        )
        sentiment_results.append({
            'news_id': news.get('id'),
            'title': news.get('title'),
            'sentiment': sentiment
        })
    
    # آمار کلی احساسات
    sentiment_counts = {
        'very_positive': 0,
        'positive': 0,
        'neutral': 0,
        'negative': 0,
        'very_negative': 0
    }
    
    for result in sentiment_results:
        sentiment_counts[result['sentiment']['overall']] += 1
    
    return {
        'total_analyzed': len(sentiment_results),
        'sentiment_distribution': sentiment_counts,
        'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get),
        'sentiment_confidence': sum(result['sentiment']['confidence'] for result in sentiment_results) / len(sentiment_results) if sentiment_results else 0,
        'detailed_results': sentiment_results[:10]  # نمونه‌ای از نتایج
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

def _analyze_advanced_sentiment(text: str) -> Dict[str, Any]:
    """تحلیل احساسات پیشرفته‌تر (قابل گسترش با ML)"""
    text_lower = text.lower()
    
    # لیست‌های گسترده‌تر کلمات (قابل گسترش)
    very_positive_words = ['surge', 'skyrocket', 'explode', 'breakout', 'record high', 'massive']
    positive_words = ['rise', 'gain', 'growth', 'bullish', 'positive', 'optimistic', 'recovery']
    negative_words = ['drop', 'fall', 'decline', 'bearish', 'negative', 'warning', 'concern']
    very_negative_words = ['crash', 'collapse', 'plummet', 'disaster', 'crisis', 'panic']
    
    very_positive = sum(1 for word in very_positive_words if word in text_lower)
    positive = sum(1 for word in positive_words if word in text_lower)
    negative = sum(1 for word in negative_words if word in text_lower)
    very_negative = sum(1 for word in very_negative_words if word in text_lower)
    
    scores = {
        'very_positive': very_positive * 2,
        'positive': positive,
        'neutral': 0,
        'negative': negative,
        'very_negative': very_negative * 2
    }
    
    max_sentiment = max(scores, key=scores.get)
    total_score = sum(scores.values())
    
    # محاسبه اطمینان
    confidence = min(total_score / 10, 1.0) if total_score > 0 else 0.0
    
    return {
        'overall': max_sentiment if total_score > 0 else 'neutral',
        'confidence': round(confidence, 2),
        'score_breakdown': scores,
        'trigger_words': _extract_trigger_words(text_lower, 
            very_positive_words + positive_words + negative_words + very_negative_words)
    }

def _extract_keywords(text: str) -> List[str]:
    """استخراج کلمات کلیدی ساده"""
    # کلمات متداول برای حذف
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = text.lower().split()
    keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    
    # شمارش تکرار
    from collections import Counter
    return [word for word, count in Counter(keywords).most_common(20)]

def _extract_trigger_words(text: str, trigger_list: List[str]) -> List[str]:
    """استخراج کلمات محرک احساسات"""
    found_triggers = []
    for trigger in trigger_list:
        if trigger in text:
            found_triggers.append(trigger)
    return found_triggers

def _is_recent(timestamp: str) -> bool:
    """بررسی جدید بودن خبر"""
    try:
        from datetime import datetime, timezone
        news_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        time_diff = current_time - news_time
        return time_diff.days < 1  # جدیدتر از 24 ساعت
    except:
        return False

def _get_category_characteristics(category: str) -> str:
    """ویژگی‌های هر دسته‌بندی خبری"""
    characteristics = {
        'handpicked': 'اخبار منتخب با اهمیت بالا',
        'trending': 'اخبار پرطرفدار و داغ',
        'latest': 'آخرین اخبار با تأکید بر تازگی',
        'bullish': 'اخبار مثبت و امیدوارکننده',
        'bearish': 'اخبار هشداردهنده و محتاطانه'
    }
    return characteristics.get(category, 'دسته‌بندی عمومی')

def _get_expected_sentiment_for_category(category: str) -> str:
    """احساسات مورد انتظار برای هر دسته‌بندی"""
    expected = {
        'bullish': 'positive',
        'bearish': 'negative',
        'handpicked': 'neutral',  # معمولاً متعادل
        'trending': 'varies',
        'latest': 'varies'
    }
    return expected.get(category, 'neutral')

def _rate_content_quality(news: Dict) -> float:
    """امتیازدهی کیفیت محتوا"""
    score = 0.0
    
    if news.get('title'):
        score += 0.3
    if news.get('description'):
        score += 0.3
    if news.get('content'):
        score += 0.4
    if news.get('url'):
        score += 0.1
    if news.get('publishedAt'):
        score += 0.1
    
    return min(score, 1.0)

def _get_content_quality_factors(news: Dict) -> List[str]:
    """عوامل کیفیت محتوا"""
    factors = []
    
    if news.get('title'):
        factors.append('has_title')
    if news.get('description'):
        factors.append('has_description')
    if news.get('content'):
        factors.append('has_content')
    if news.get('url'):
        factors.append('has_source_url')
    if news.get('publishedAt'):
        factors.append('has_timestamp')
    if news.get('tags'):
        factors.append('has_tags')
    
    return factors

def _get_news_field_descriptions() -> Dict[str, str]:
    """توضیحات فیلدهای خبری"""
    return {
        'id': 'شناسه یکتای خبر',
        'title': 'عنوان خبر',
        'description': 'خلاصه یا چکیده خبر',
        'content': 'محتوای کامل خبر',
        'url': 'لینک منبع اصلی',
        'source': 'نام منبع خبر',
        'publishedAt': 'زمان انتشار خبر',
        'author': 'نویسنده خبر',
        'tags': 'تگ‌های موضوعی مرتبط',
        'coverage': 'حوزه پوشش خبری'
    }
