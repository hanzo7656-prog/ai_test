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
    """دریافت اخبار پردازش شده عمومی از CoinStats API"""
    try:
        logger.info(f"📰 Fetching news - Limit: {limit}, Page: {page}")
        
        raw_data = coin_stats_manager.get_news(limit=limit)
        
        if "error" in raw_data:
            logger.error(f"❌ News API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        news_items = raw_data.get('data', [])
        
        # پردازش و آنالیز اخبار
        processed_news = []
        for news_item in news_items:
            processed_news.append({
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'description': news_item.get('description'),
                'url': news_item.get('url'),
                'source': news_item.get('source'),
                'published_at': news_item.get('published_at', news_item.get('publishedAt')),
                'image_url': news_item.get('imageUrl'),
                'sentiment': _analyze_sentiment(news_item),
                'importance_score': _calculate_importance_score(news_item),
                'reliability_score': _calculate_reliability_score(news_item),
                'tags': news_item.get('tags', []),
                'categories': news_item.get('categories', []),
                'last_updated': datetime.now().isoformat()
            })
        
        # تحلیل کلی مجموعه اخبار
        news_analysis = _analyze_news_collection(processed_news)
        
        response = {
            'status': 'success',
            'data': processed_news,
            'meta': {
                'total': len(processed_news),
                'limit': limit,
                'page': page,
                'analysis': news_analysis
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
    limit: int = Query(10, ge=1, le=50, description="تعداد اخبار (۱ تا ۵۰)"),
    page: int = Query(1, ge=1, description="شماره صفحه")
):
    """دریافت اخبار پردازش شده - پشتیبانی از انواع مختلف"""
    try:
        # اعتبارسنجی نوع خبر
        valid_types = ["latest", "trending", "featured", "breaking", "analysis"]
        if news_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid news type. Valid types: {valid_types}")
        
        logger.info(f"📰 Fetching {news_type} news - Limit: {limit}")
        
        raw_data = coin_stats_manager.get_news_by_type(news_type, limit=limit)
        
        if "error" in raw_data:
            logger.error(f"❌ {news_type} news API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        news_items = raw_data.get('data', [])
        
        processed_news = []
        for news_item in news_items:
            processed_news.append({
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'description': news_item.get('description'),
                'url': news_item.get('url'),
                'source': news_item.get('source'),
                'published_at': news_item.get('published_at', news_item.get('publishedAt')),
                'image_url': news_item.get('imageUrl'),
                'type': news_type,
                'sentiment': _analyze_sentiment(news_item),
                'importance_score': _calculate_importance_score(news_item),
                'reliability_score': _calculate_reliability_score(news_item),
                'tags': news_item.get('tags', []),
                'last_updated': datetime.now().isoformat()
            })
        
        response = {
            'status': 'success',
            'data': processed_news,
            'meta': {
                'type': news_type,
                'total': len(processed_news),
                'limit': limit,
                'page': page
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
    """دریافت لیست منابع خبری معتبر"""
    try:
        logger.info("📰 Fetching news sources")
        
        raw_data = coin_stats_manager.get_news_sources()
        
        if "error" in raw_data:
            logger.error(f"❌ News sources API error: {raw_data['error']}")
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        sources = raw_data.get('data', [])
        
        processed_sources = []
        for source in sources:
            reliability_score = _calculate_source_reliability(source)
            processed_sources.append({
                'id': source.get('id'),
                'name': source.get('name'),
                'url': source.get('url'),
                'description': source.get('description'),
                'language': source.get('language', 'en'),
                'country': source.get('country'),
                'category': source.get('category', 'crypto'),
                'reliability_score': reliability_score,
                'coverage': source.get('coverage', 'general'),
                'last_updated': datetime.now().isoformat()
            })
        
        # مرتب‌سازی بر اساس قابلیت اطمینان
        processed_sources.sort(key=lambda x: x['reliability_score'], reverse=True)
        
        response = {
            'status': 'success',
            'data': processed_sources,
            'meta': {
                'total': len(processed_sources),
                'high_reliability_sources': len([s for s in processed_sources if s['reliability_score'] >= 4])
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
        
        news_data = raw_data.get('data', {})
        
        # پردازش پیشرفته جزئیات خبر
        processed_detail = {
            'id': news_data.get('id'),
            'title': news_data.get('title'),
            'content': news_data.get('content', news_data.get('description')),
            'summary': _generate_advanced_summary(news_data),
            'url': news_data.get('url'),
            'source': news_data.get('source'),
            'author': news_data.get('author'),
            'published_at': news_data.get('published_at', news_data.get('publishedAt')),
            'image_url': news_data.get('imageUrl'),
            'sentiment': _analyze_sentiment(news_data),
            'importance_score': _calculate_importance_score(news_data),
            'reliability_score': _calculate_reliability_score(news_data),
            'key_points': _extract_key_points(news_data),
            'tags': news_data.get('tags', []),
            'categories': news_data.get('categories', []),
            'related_coins': _extract_related_coins(news_data),
            'reading_time': _estimate_reading_time(news_data),
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

@news_router.get("/analysis/sentiment", summary="تحلیل احساسات اخبار")
async def get_news_sentiment_analysis(
    limit: int = Query(20, ge=1, le=100, description="تعداد اخبار برای تحلیل")
):
    """تحلیل پیشرفته احساسات مجموعه اخبار"""
    try:
        logger.info(f"📊 Analyzing news sentiment - Limit: {limit}")
        
        raw_data = coin_stats_manager.get_news(limit=limit)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        news_items = raw_data.get('data', [])
        
        # تحلیل احساسات برای هر خبر
        sentiment_analysis = []
        for news_item in news_items:
            sentiment_data = {
                'id': news_item.get('id'),
                'title': news_item.get('title'),
                'sentiment': _analyze_sentiment(news_item),
                'confidence': _calculate_sentiment_confidence(news_item),
                'keywords': _extract_sentiment_keywords(news_item),
                'impact_score': _calculate_impact_score(news_item)
            }
            sentiment_analysis.append(sentiment_data)
        
        # تحلیل کلی احساسات
        overall_sentiment = _calculate_overall_sentiment(sentiment_analysis)
        
        response = {
            'status': 'success',
            'data': sentiment_analysis,
            'analysis': {
                'overall_sentiment': overall_sentiment['sentiment'],
                'sentiment_distribution': overall_sentiment['distribution'],
                'average_confidence': overall_sentiment['average_confidence'],
                'total_analyzed': len(sentiment_analysis),
                'market_outlook': overall_sentiment['market_outlook']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Sentiment analysis completed - Overall: {overall_sentiment['sentiment']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@news_router.get("/trending/topics", summary="موضوعات داغ")
async def get_trending_topics(
    limit: int = Query(10, ge=1, le=50, description="تعداد موضوعات")
):
    """استخراج موضوعات داغ و ترند از اخبار"""
    try:
        logger.info(f"🔥 Extracting trending topics - Limit: {limit}")
        
        raw_data = coin_stats_manager.get_news(limit=50)  # اخبار بیشتر برای تحلیل بهتر
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        news_items = raw_data.get('data', [])
        
        # استخراج و تحلیل موضوعات
        trending_topics = _extract_trending_topics(news_items, limit)
        
        response = {
            'status': 'success',
            'data': trending_topics,
            'meta': {
                'total_topics': len(trending_topics),
                'analysis_period': 'recent',
                'sources_analyzed': len(news_items)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Trending topics extracted - Total: {len(trending_topics)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error in trending topics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================ توابع کمکی پیشرفته ============================

def _analyze_sentiment(news_item: Dict) -> str:
    """تحلیل پیشرفته احساسات خبر با الگوریتم بهبود یافته"""
    title = news_item.get('title', '').lower()
    description = news_item.get('description', '').lower()
    content = f"{title} {description}"
    
    if not content.strip():
        return "neutral"
    
    # دیکشنری‌های احساسات پیشرفته
    positive_words = {
        'bullish': 3, 'surge': 3, 'rally': 3, 'gain': 2, 'positive': 2, 
        'growth': 2, 'soar': 3, 'moon': 3, 'breakout': 2, 'uptrend': 2,
        'profit': 2, 'success': 2, 'adoption': 2, 'innovation': 1, 'partnership': 1
    }
    
    negative_words = {
        'bearish': 3, 'drop': 3, 'crash': 3, 'loss': 2, 'negative': 2, 
        'decline': 2, 'suffer': 2, 'dump': 3, 'plunge': 3, 'downtrend': 2,
        'risk': 2, 'warning': 2, 'concern': 1, 'volatility': 1, 'regulation': 1
    }
    
    # محاسبه امتیاز
    positive_score = sum(score for word, score in positive_words.items() if word in content)
    negative_score = sum(score for word, score in negative_words.items() if word in content)
    
    # تحلیل بر اساس اختلاف امتیاز
    score_diff = positive_score - negative_score
    
    if score_diff >= 3:
        return "strongly_bullish"
    elif score_diff >= 1:
        return "bullish"
    elif score_diff <= -3:
        return "strongly_bearish"
    elif score_diff <= -1:
        return "bearish"
    else:
        return "neutral"

def _calculate_importance_score(news_item: Dict) -> int:
    """محاسبه امتیاز اهمیت خبر (۰-۱۰)"""
    score = 0
    
    # منبع معتبر
    reliable_sources = {
        'cointelegraph': 3, 'decrypt': 3, 'coindesk': 3, 'bloomberg': 4,
        'reuters': 4, 'benzinga': 2, 'newsbtc': 2, 'cryptopotato': 2
    }
    
    source = news_item.get('source', '').lower()
    for rel_source, points in reliable_sources.items():
        if rel_source in source:
            score += points
            break
    
    # طول عنوان (عنوان‌های طولانی‌تر معمولاً مهم‌ترند)
    title_length = len(news_item.get('title', ''))
    if title_length > 80:
        score += 2
    elif title_length > 50:
        score += 1
    
    # وجود توضیحات کامل
    if news_item.get('description') and len(news_item['description']) > 100:
        score += 2
    
    # تگ‌های مهم
    important_tags = ['bitcoin', 'ethereum', 'regulation', 'adoption', 'defi', 'nft']
    tags = news_item.get('tags', [])
    if any(tag in important_tags for tag in tags):
        score += 2
    
    return min(score, 10)

def _calculate_reliability_score(news_item: Dict) -> int:
    """محاسبه قابلیت اطمینان خبر (۱-۵)"""
    source = news_item.get('source', '').lower()
    
    reliability_scores = {
        'cointelegraph': 5, 'decrypt': 4, 'coindesk': 4, 'bloomberg': 5,
        'reuters': 5, 'benzinga': 3, 'newsbtc': 3, 'cryptopotato': 3,
        'dailyhodl': 3, 'cryptoslate': 3
    }
    
    for rel_source, score in reliability_scores.items():
        if rel_source in source:
            return score
    
    return 2  # پیش‌فرض برای منابع ناشناخته

def _calculate_source_reliability(source: Dict) -> int:
    """محاسبه قابلیت اطمینان منبع"""
    source_name = source.get('name', '').lower()
    
    reliability_scores = {
        'cointelegraph': 5, 'decrypt': 4, 'coindesk': 4, 'bloomberg': 5,
        'reuters': 5, 'benzinga': 3, 'newsbtc': 3, 'cryptopotato': 3
    }
    
    for rel_source, score in reliability_scores.items():
        if rel_source in source_name:
            return score
    
    return 2

def _generate_advanced_summary(news_item: Dict) -> str:
    """تولید خلاصه پیشرفته خبر"""
    content = news_item.get('content') or news_item.get('description') or news_item.get('title', '')
    
    if not content:
        return "No summary available"
    
    # خلاصه‌سازی ساده (در نسخه واقعی از NLP استفاده می‌شود)
    if len(content) > 200:
        return content[:197] + '...'
    return content

def _extract_key_points(news_item: Dict) -> List[str]:
    """استخراج نکات کلیدی از خبر"""
    content = news_item.get('content') or news_item.get('description') or ''
    title = news_item.get('title', '')
    
    if not content and not title:
        return ["No key points available"]
    
    # استخراج جملات کلیدی ساده
    sentences = content.split('.')
    key_points = []
    
    # اضافه کردن عنوان به عنوان نکته اول
    if title:
        key_points.append(title)
    
    # اضافه کردن ۲-۳ جمله اول به عنوان نکات کلیدی
    for sentence in sentences[:3]:
        sentence = sentence.strip()
        if len(sentence) > 20 and sentence not in key_points:
            key_points.append(sentence)
    
    return key_points if key_points else [content[:100] + '...' if content else "No content"]

def _extract_related_coins(news_item: Dict) -> List[str]:
    """استخراج ارزهای مرتبط از خبر"""
    content = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
    
    crypto_keywords = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'cardano', 'ada',
        'binance', 'bnb', 'ripple', 'xrp', 'polkadot', 'dot', 'dogecoin', 'doge'
    ]
    
    related_coins = []
    for coin in crypto_keywords:
        if coin in content:
            related_coins.append(coin)
    
    return list(set(related_coins))  # حذف موارد تکراری

def _estimate_reading_time(news_item: Dict) -> str:
    """تخمین زمان مطالعه خبر"""
    content = news_item.get('content') or news_item.get('description') or ''
    word_count = len(content.split())
    
    # فرض: ۲۰۰ کلمه در دقیقه
    minutes = max(1, round(word_count / 200))
    return f"{minutes} min"

def _analyze_news_collection(news_items: List[Dict]) -> Dict[str, Any]:
    """تحلیل کلی مجموعه اخبار"""
    if not news_items:
        return {"message": "No news to analyze"}
    
    sentiment_count = {
        'strongly_bullish': 0,
        'bullish': 0, 
        'neutral': 0,
        'bearish': 0,
        'strongly_bearish': 0
    }
    
    total_importance = 0
    total_reliability = 0
    
    for news in news_items:
        sentiment = news.get('sentiment', 'neutral')
        sentiment_count[sentiment] = sentiment_count.get(sentiment, 0) + 1
        total_importance += news.get('importance_score', 0)
        total_reliability += news.get('reliability_score', 0)
    
    # محاسبه میانگین‌ها
    avg_importance = total_importance / len(news_items)
    avg_reliability = total_reliability / len(news_items)
    
    # تعیین احساسات غالب
    dominant_sentiment = max(sentiment_count.items(), key=lambda x: x[1])[0]
    
    return {
        'total_news': len(news_items),
        'sentiment_distribution': sentiment_count,
        'dominant_sentiment': dominant_sentiment,
        'average_importance': round(avg_importance, 2),
        'average_reliability': round(avg_reliability, 2),
        'high_importance_news': len([n for n in news_items if n.get('importance_score', 0) >= 7]),
        'high_reliability_news': len([n for n in news_items if n.get('reliability_score', 0) >= 4])
    }

def _calculate_sentiment_confidence(news_item: Dict) -> float:
    """محاسبه میزان اطمینان تحلیل احساسات"""
    content = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
    
    if not content.strip():
        return 0.5
    
    # محاسبه بر اساس تعداد کلمات کلیدی
    positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'positive', 'growth', 'soar']
    negative_keywords = ['bearish', 'drop', 'crash', 'loss', 'negative', 'decline', 'suffer']
    
    positive_count = sum(1 for word in positive_keywords if word in content)
    negative_count = sum(1 for word in negative_keywords if word in content)
    total_keywords = positive_count + negative_count
    
    if total_keywords == 0:
        return 0.3  # اطمینان پایین برای اخبار خنثی
    
    return min(total_keywords / 10, 0.9)  # نرمال‌سازی به ۰-۰.۹

def _extract_sentiment_keywords(news_item: Dict) -> List[str]:
    """استخراج کلمات کلیدی احساسات"""
    content = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
    
    sentiment_keywords = [
        'bullish', 'bearish', 'surge', 'crash', 'rally', 'drop', 
        'gain', 'loss', 'positive', 'negative', 'growth', 'decline'
    ]
    
    return [word for word in sentiment_keywords if word in content]

def _calculate_impact_score(news_item: Dict) -> int:
    """محاسبه امتیاز تاثیر خبر"""
    score = news_item.get('importance_score', 0) + news_item.get('reliability_score', 0)
    return min(score, 10)

def _calculate_overall_sentiment(sentiment_data: List[Dict]) -> Dict[str, Any]:
    """محاسبه احساسات کلی مجموعه اخبار"""
    sentiment_distribution = {
        'strongly_bullish': 0,
        'bullish': 0,
        'neutral': 0, 
        'bearish': 0,
        'strongly_bearish': 0
    }
    
    total_confidence = 0
    
    for item in sentiment_data:
        sentiment = item.get('sentiment', 'neutral')
        sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
        total_confidence += item.get('confidence', 0)
    
    # تعیین احساسات غالب
    dominant_sentiment = max(sentiment_distribution.items(), key=lambda x: x[1])[0]
    avg_confidence = total_confidence / len(sentiment_data) if sentiment_data else 0
    
    # تحلیل بازار بر اساس احساسات
    bull_count = sentiment_distribution['strongly_bullish'] + sentiment_distribution['bullish']
    bear_count = sentiment_distribution['strongly_bearish'] + sentiment_distribution['bearish']
    
    if bull_count > bear_count * 1.5:
        market_outlook = "bullish"
    elif bear_count > bull_count * 1.5:
        market_outlook = "bearish" 
    else:
        market_outlook = "neutral"
    
    return {
        'sentiment': dominant_sentiment,
        'distribution': sentiment_distribution,
        'average_confidence': round(avg_confidence, 2),
        'market_outlook': market_outlook
    }

def _extract_trending_topics(news_items: List[Dict], limit: int = 10) -> List[Dict]:
    """استخراج موضوعات داغ و ترند"""
    from collections import Counter
    
    # استخراج کلمات کلیدی از عنوان‌ها
    all_keywords = []
    crypto_terms = [
        'bitcoin', 'ethereum', 'defi', 'nft', 'web3', 'metaverse', 'dao',
        'layer2', 'scaling', 'regulation', 'adoption', 'institutional',
        'bull market', 'bear market', 'halving', 'mining', 'staking'
    ]
    
    for news in news_items:
        title = news.get('title', '').lower()
        # اضافه کردن کلمات کلیدی مرتبط
        for term in crypto_terms:
            if term in title:
                all_keywords.append(term)
    
    # شمارش و مرتب‌سازی
    topic_counter = Counter(all_keywords)
    trending_topics = []
    
    for topic, count in topic_counter.most_common(limit):
        # محاسبه شدت ترند
        intensity = min(count / len(news_items) * 100, 100)
        
        trending_topics.append({
            'topic': topic,
            'frequency': count,
            'intensity': round(intensity, 1),
            'trend_level': 'high' if intensity > 30 else 'medium' if intensity > 15 else 'low'
        })
    
    return trending_topics
