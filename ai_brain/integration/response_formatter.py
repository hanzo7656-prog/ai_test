import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """فرمت‌دهنده پاسخ‌های هوش مصنوعی به زبان طبیعی"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get('language', 'fa')
        
        # الگوهای فرمت‌دهی بر اساس intent
        self.response_templates = self._initialize_templates()
        
        # نمادها و ایموجی‌ها
        self.symbols = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'bitcoin': '₿',
            'ethereum': 'Ξ',
            'up': '📈',
            'down': '📉',
            'stable': '➡️',
            'news': '📰',
            'health': '🏥',
            'cache': '💾',
            'alert': '🚨',
            'list': '📋',
            'chart': '📊'
        }
        
        logger.info("🚀 فرمت‌دهنده پاسخ راه‌اندازی شد")
    
    def _initialize_templates(self) -> Dict[str, Any]:
        """مقداردهی اولیه الگوهای پاسخ"""
        return {
            'health_check': {
                'fa': "🏥 وضعیت سیستم: {status}\n• امتیاز سلامت: {health_score}%\n• وضعیت: {system_status}",
                'en': "🏥 System Status: {status}\n• Health Score: {health_score}%\n• Status: {system_status}"
            },
            'cache_status': {
                'fa': "💾 سیستم کش: {connected_dbs}/5 دیتابیس متصل\n• امتیاز: {health_score}%\n• حافظه استفاده شده: {used_memory}MB",
                'en': "💾 Cache System: {connected_dbs}/5 databases connected\n• Score: {health_score}%\n• Memory Used: {used_memory}MB"
            },
            'price_request': {
                'fa': "{symbol} {coin_name}: ${price:,.2f}\n• تغییر 24h: {trend} {change_percent:.2f}%\n• حجم معاملات: ${volume:,.0f}",
                'en': "{symbol} {coin_name}: ${price:,.2f}\n• 24h Change: {trend} {change_percent:.2f}%\n• Volume: ${volume:,.0f}"
            },
            'list_request': {
                'fa': "🏆 {count} ارز برتر:\n{coins_list}",
                'en': "🏆 Top {count} coins:\n{coins_list}"
            },
            'news_request': {
                'fa': "📰 {count} خبر جدید:\n{news_list}",
                'en': "📰 {count} news items:\n{news_list}"
            },
            'fear_greed': {
                'fa': "😨😊 شاخص ترس و طمع: {value}/100\n• وضعیت: {classification}\n• تحلیل: {analysis}",
                'en': "😨😊 Fear & Greed Index: {value}/100\n• Status: {classification}\n• Analysis: {analysis}"
            },
            'error': {
                'fa': "❌ خطا در دریافت اطلاعات: {error}\n• لطفاً دوباره تلاش کنید",
                'en': "❌ Error retrieving data: {error}\n• Please try again"
            },
            'capacity_error': {
                'fa': "⚠️ پتانسیل پردازش این سوال را ندارم.\n• لطفاً سوال ساده‌تری مطرح کنید",
                'en': "⚠️ I don't have the capacity to process this question.\n• Please ask a simpler question"
            },
            'unknown_intent': {
                'fa': "🤔 متوجه سوال شما نشدم.\n• می‌توانید در مورد این موارد بپرسید: سلامت سیستم، قیمت ارزها، اخبار، وضعیت کش",
                'en': "🤔 I didn't understand your question.\n• You can ask about: system health, coin prices, news, cache status"
            }
        }
    
    def format_response(self, intent: str, api_data: Dict[str, Any], user_language: str = 'fa') -> str:
        """فرمت‌دهی پاسخ بر اساس intent و داده API"""
        
        if not api_data.get('success', False):
            error_msg = api_data.get('error', 'خطای ناشناخته')
            return self._format_error_response(error_msg, user_language)
        
        data = api_data.get('data', {})
        
        try:
            if intent == 'health_check':
                return self._format_health_response(data, user_language)
            elif intent == 'cache_status':
                return self._format_cache_response(data, user_language)
            elif intent == 'price_request':
                return self._format_price_response(data, user_language)
            elif intent == 'list_request':
                return self._format_list_response(data, user_language)
            elif intent == 'news_request':
                return self._format_news_response(data, user_language)
            elif intent == 'fear_greed':
                return self._format_fear_greed_response(data, user_language)
            elif intent == 'alerts_status':
                return self._format_alerts_response(data, user_language)
            elif intent == 'metrics_status':
                return self._format_metrics_response(data, user_language)
            else:
                return self._format_generic_response(intent, data, user_language)
                
        except Exception as e:
            logger.error(f"❌ خطا در فرمت‌دهی پاسخ برای {intent}: {e}")
            return self._format_error_response("خطا در پردازش پاسخ", user_language)
    
    def _format_health_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ سلامت سیستم"""
        health_score = data.get('health_score', 0)
        status = data.get('status', 'unknown')
        
        status_emoji = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
        status_text = "عالی" if health_score > 80 else "قابل قبول" if health_score > 60 else "نیاز توجه"
        
        template = self.response_templates['health_check'][language]
        return template.format(
            status=f"{status_emoji} {status_text}",
            health_score=health_score,
            system_status=status
        )
    
    def _format_cache_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ وضعیت کش"""
        cache_health = data.get('health', {})
        connected_dbs = cache_health.get('cloud_resources', {}).get('databases_connected', 0)
        health_score = cache_health.get('health_score', 0)
        used_memory = cache_health.get('cloud_resources', {}).get('storage_used_mb', 0)
        
        template = self.response_templates['cache_status'][language]
        return template.format(
            connected_dbs=connected_dbs,
            health_score=health_score,
            used_memory=used_memory
        )
    
    def _format_price_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ قیمت"""
        coin_data = data.get('data', {})
        
        coin_name = coin_data.get('name', 'Unknown')
        symbol = coin_data.get('symbol', '').upper()
        price = coin_data.get('price', 0)
        change_24h = coin_data.get('price_change_24h', 0)
        volume = coin_data.get('volume_24h', 0)
        
        # تشخیص روند
        if change_24h > 0:
            trend = self.symbols['up']
        elif change_24h < 0:
            trend = self.symbols['down']
        else:
            trend = self.symbols['stable']
        
        # نماد اختصاصی برای ارزهای معروف
        coin_symbol = self.symbols.get(coin_name.lower(), f"{symbol}")
        
        template = self.response_templates['price_request'][language]
        return template.format(
            symbol=coin_symbol,
            coin_name=coin_name,
            price=price,
            trend=trend,
            change_percent=abs(change_24h),
            volume=volume
        )
    
    def _format_list_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ لیست ارزها"""
        coins = data.get('data', [])
        count = len(coins)
        
        if count == 0:
            return "❌ اطلاعات ارزی دریافت نشد"
        
        # ساخت لیست ارزها
        coins_list = ""
        for i, coin in enumerate(coins[:5]):  # حداکثر 5 ارز
            name = coin.get('name', 'Unknown')
            symbol = coin.get('symbol', '').upper()
            price = coin.get('price', 0)
            
            coins_list += f"{i+1}. {symbol}: ${price:,.2f}\n"
        
        template = self.response_templates['list_request'][language]
        return template.format(count=count, coins_list=coins_list.strip())
    
    def _format_news_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ اخبار"""
        news_items = data.get('data', [])
        count = len(news_items)
        
        if count == 0:
            return "📰 خبری یافت نشد"
        
        # ساخت لیست اخبار
        news_list = ""
        for i, news in enumerate(news_items[:3]):  # حداکثر 3 خبر
            title = news.get('title', 'بدون عنوان')
            source = news.get('source', 'منبع ناشناس')
            
            # کوتاه کردن عنوان اگر طولانی باشد
            if len(title) > 60:
                title = title[:57] + "..."
            
            news_list += f"• {title} ({source})\n"
        
        template = self.response_templates['news_request'][language]
        return template.format(count=count, news_list=news_list.strip())
    
    def _format_fear_greed_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ شاخص ترس و طمع"""
        fear_data = data.get('data', {})
        
        value = fear_data.get('value', 50)
        classification = fear_data.get('value_classification', 'Neutral')
        
        # تحلیل بر اساس مقدار
        if value >= 75:
            analysis = "احتیاط - بازار ممکن است overbought باشد"
        elif value >= 55:
            analysis = "مناسب برای نگهداری"
        elif value >= 45:
            analysis = "متعادل - فرصت‌های خوب"
        elif value >= 25:
            analysis = "مناسب برای خرید"
        else:
            analysis = "فرصت عالی - بازار oversold است"
        
        template = self.response_templates['fear_greed'][language]
        return template.format(
            value=value,
            classification=classification,
            analysis=analysis
        )
    
    def _format_alerts_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ هشدارها"""
        active_alerts = data.get('active_alerts', [])
        
        if not active_alerts:
            return "✅ هیچ هشدار فعالی وجود ندارد"
        
        critical_count = len([a for a in active_alerts if a.get('level') == 'CRITICAL'])
        warning_count = len([a for a in active_alerts if a.get('level') == 'WARNING'])
        
        if language == 'fa':
            return f"🚨 {len(active_alerts)} هشدار فعال\n• 🔴 {critical_count} هشدار بحرانی\n• 🟡 {warning_count} هشدار هشدار"
        else:
            return f"🚨 {len(active_alerts)} active alerts\n• 🔴 {critical_count} critical\n• 🟡 {warning_count} warnings"
    
    def _format_metrics_response(self, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ متریک‌ها"""
        system_metrics = data.get('system', {})
        
        cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
        memory_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
        
        if language == 'fa':
            return f"📊 مصرف منابع:\n• پردازنده: {cpu_usage}%\n• حافظه: {memory_usage}%"
        else:
            return f"📊 Resource Usage:\n• CPU: {cpu_usage}%\n• Memory: {memory_usage}%"
    
    def _format_generic_response(self, intent: str, data: Dict[str, Any], language: str) -> str:
        """فرمت‌دهی پاسخ عمومی"""
        if language == 'fa':
            return f"📊 اطلاعات دریافت شد: {intent}\n• داده‌ها با موفقیت پردازش شدند"
        else:
            return f"📊 Data received: {intent}\n• Information processed successfully"
    
    def _format_error_response(self, error_message: str, language: str) -> str:
        """فرمت‌دهی پاسخ خطا"""
        template = self.response_templates['error'][language]
        return template.format(error=error_message)
    
    def format_capacity_error(self, user_language: str = 'fa') -> str:
        """فرمت‌دهی خطای ظرفیت پردازش"""
        template = self.response_templates['capacity_error'][user_language]
        return template
    
    def format_unknown_intent(self, user_language: str = 'fa') -> str:
        """فرمت‌دهی پاسخ برای intent ناشناخته"""
        template = self.response_templates['unknown_intent'][user_language]
        return template
    
    def detect_user_language(self, user_input: str) -> str:
        """تشخیص زبان کاربر از متن ورودی"""
        # آنالیز ساده بر اساس کاراکترها
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', user_input))
        english_chars = len(re.findall(r'[a-zA-Z]', user_input))
        
        if persian_chars > english_chars:
            return 'fa'
        else:
            return 'en'
    
    def get_response_stats(self) -> Dict[str, Any]:
        """آمار فرمت‌دهی پاسخ"""
        return {
            'supported_intents': len(self.response_templates),
            'default_language': self.language,
            'symbols_count': len(self.symbols)
        }
