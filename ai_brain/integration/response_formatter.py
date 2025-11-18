# ai_brain/integration/response_formatter.py
class ResponseFormatter:
    def __init__(self, config: dict):
        self.config = config
        self.response_templates = self._load_templates()
    
    def _load_templates(self) -> dict:
        """بارگذاری تمپلیت‌های پاسخ"""
        return {
            "fa": {
                "price_check": "💰 قیمت {symbol}: ${price:,.2f} ({change:+.2f}%) - حجم: ${volume:,.0f}",
                "system_status": "🖥️ وضعیت سیستم:\n• CPU: {cpu_usage}%\n• حافظه: {memory_usage}%\n• دیسک: {disk_usage}%\n• آپتایم: {uptime}",
                "news_request": "📰 آخرین اخبار {category}:\n{articles}",
                "technical_analysis": "📊 تحلیل تکنیکال {symbol}:\n• RSI: {rsi}\n• MACD: {macd}\n• حمایت: {support}\n• مقاومت: {resistance}",
                "fear_greed": "😨📈 شاخص ترس و طمع: {index}/100\n• وضعیت: {status}",
                "market_summary": "📈 خلاصه بازار:\n• حجم کل: ${total_volume:,.0f}\n• ارزهای صعودی: {gainers}\n• ارزهای نزولی: {losers}",
                "ai_analysis": "🤖 تحلیل هوش مصنوعی:\n{analysis}",
                "error": "❌ خطا: {message}",
                "success": "✅ {message}",
                "processing": "⏳ در حال پردازش...",
                "no_data": "📭 داده‌ای یافت نشد"
            },
            "en": {
                "price_check": "💰 Price {symbol}: ${price:,.2f} ({change:+.2f}%) - Volume: ${volume:,.0f}",
                "system_status": "🖥️ System Status:\n• CPU: {cpu_usage}%\n• Memory: {memory_usage}%\n• Disk: {disk_usage}%\n• Uptime: {uptime}",
                "news_request": "📰 Latest {category} News:\n{articles}",
                "technical_analysis": "📊 Technical Analysis {symbol}:\n• RSI: {rsi}\n• MACD: {macd}\n• Support: {support}\n• Resistance: {resistance}",
                "fear_greed": "😨📈 Fear & Greed Index: {index}/100\n• Status: {status}",
                "market_summary": "📈 Market Summary:\n• Total Volume: ${total_volume:,.0f}\n• Gainers: {gainers}\n• Losers: {losers}",
                "ai_analysis": "🤖 AI Analysis:\n{analysis}",
                "error": "❌ Error: {message}",
                "success": "✅ {message}",
                "processing": "⏳ Processing...",
                "no_data": "📭 No data found"
            }
        }
    
    def format_error_response(self, error_message: str, error_type: str = "processing_error") -> str:
        """فرمت‌بندی پاسخ خطا"""
        error_templates = {
            "processing_error": "❌ خطا در پردازش درخواست: {}",
            "api_error": "🌐 خطا در ارتباط با سرویس: {}",
            "capacity_error": "⚡ سیستم در حال حاضر ظرفیت پردازش ندارد",
            "network_error": "📡 خطا در ارتباط شبکه",
            "timeout_error": "⏰ زمان پردازش به پایان رسید",
            "authentication_error": "🔐 خطای احراز هویت",
            "rate_limit_error": "🚫 محدودیت درخواست - لطفاً کمی صبر کنید",
            "internal_error": "🔧 خطای داخلی سیستم: {}"
        }
        
        template = error_templates.get(error_type, error_templates["processing_error"])
        return template.format(error_message)
    
    def format_capacity_error(self) -> str:
        """فرمت‌بندی خطای ظرفیت"""
        return self.format_error_response("", "capacity_error")
    
    def detect_user_language(self, text: str) -> str:
        """تشخیص زبان کاربر با الگوریتم پیشرفته"""
        if not text:
            return "fa"
        
        # تشخیص بر اساس کاراکترهای فارسی/عربی
        persian_arabic_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیةيك')
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        fa_count = sum(1 for char in text if char in persian_arabic_chars)
        en_count = sum(1 for char in text if char in english_chars)
        
        # کلمات کلیدی فارسی
        persian_keywords = ['سلام', 'خداحافظ', 'لطفا', 'بله', 'خیر', 'چطور', 'چگونه', 'قیمت', 'وضعیت']
        fa_keyword_count = sum(1 for keyword in persian_keywords if keyword in text)
        
        if fa_count > en_count or fa_keyword_count > 0:
            return "fa"
        return "en"
    
    def format_response(self, intent: str, api_response: dict, user_language: str = "fa") -> str:
        """فرمت‌بندی پاسخ اصلی با قابلیت‌های پیشرفته"""
        try:
            if not api_response.get('success', False):
                error_msg = api_response.get('error', 'خطای ناشناخته')
                return self.format_error_response(error_msg, "api_error")
            
            data = api_response.get('data', {})
            
            # فرمت‌بندی هوشمند بر اساس intent و داده‌های موجود
            if intent == "price_check":
                return self._format_price_response(data, user_language)
            elif intent == "system_status":
                return self._format_system_response(data, user_language)
            elif intent == "news_request":
                return self._format_news_response(data, user_language)
            elif intent == "technical_analysis":
                return self._format_technical_analysis(data, user_language)
            elif intent == "fear_greed_index":
                return self._format_fear_greed(data, user_language)
            elif intent == "market_summary":
                return self._format_market_summary(data, user_language)
            elif intent == "ai_analysis":
                return self._format_ai_analysis(data, user_language)
            else:
                return self._format_general_response(data, user_language, intent)
                
        except Exception as e:
            return self.format_error_response(f"خطا در فرمت‌بندی پاسخ: {str(e)}")
    
    def _format_price_response(self, data: dict, language: str) -> str:
        """فرمت‌بندی پاسخ قیمت با جزئیات کامل"""
        template = self.response_templates[language]["price_check"]
        
        symbol = data.get('symbol', 'نامشخص')
        price = data.get('price', 0)
        change = data.get('change_24h', 0)
        volume = data.get('volume_24h', 0)
        high_24h = data.get('high_24h', 0)
        low_24h = data.get('low_24h', 0)
        
        # اضافه کردن اطلاعات اضافی
        additional_info = ""
        if language == "fa":
            if high_24h and low_24h:
                additional_info = f"\n📊 دامنه 24h: ${low_24h:,.2f} - ${high_24h:,.2f}"
        else:
            if high_24h and low_24h:
                additional_info = f"\n📊 24h Range: ${low_24h:,.2f} - ${high_24h:,.2f}"
        
        return template.format(
            symbol=symbol,
            price=price,
            change=change,
            volume=volume
        ) + additional_info
    
    def _format_system_response(self, data: dict, language: str) -> str:
        """فرمت‌بندی پاسخ وضعیت سیستم"""
        template = self.response_templates[language]["system_status"]
        
        cpu_usage = data.get('cpu_usage', 0)
        memory_usage = data.get('memory_usage', 0)
        disk_usage = data.get('disk_usage', 0)
        uptime = data.get('uptime', 'نامشخص')
        active_connections = data.get('active_connections', 0)
        
        # اضافه کردن اطلاعات شبکه
        network_info = ""
        if language == "fa":
            network_info = f"\n• اتصالات فعال: {active_connections}"
        else:
            network_info = f"\n• Active Connections: {active_connections}"
        
        return template.format(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            uptime=uptime
        ) + network_info
    
    def _format_news_response(self, data: dict, language: str) -> str:
        """فرمت‌بندی پاسخ اخبار"""
        articles = data.get('articles', [])
        category = data.get('category', '')
        
        if not articles:
            return self.response_templates[language]["no_data"]
        
        articles_text = ""
        for i, article in enumerate(articles[:5]):  # حداکثر 5 خبر
            title = article.get('title', 'بدون عنوان')
            source = article.get('source', '')
            published_at = article.get('published_at', '')
            
            if language == "fa":
                articles_text += f"{i+1}. {title}"
                if source:
                    articles_text += f" ({source})"
                articles_text += "\n"
            else:
                articles_text += f"{i+1}. {title}"
                if source:
                    articles_text += f" ({source})"
                articles_text += "\n"
        
        template = self.response_templates[language]["news_request"]
        return template.format(category=category, articles=articles_text)
    
    def _format_technical_analysis(self, data: dict, language: str) -> str:
        """فرمت‌بندی تحلیل تکنیکال"""
        template = self.response_templates[language]["technical_analysis"]
        
        symbol = data.get('symbol', 'نامشخص')
        rsi = data.get('rsi', 'N/A')
        macd = data.get('macd', 'N/A')
        support = data.get('support_levels', ['N/A'])[0]
        resistance = data.get('resistance_levels', ['N/A'])[0]
        trend = data.get('trend', 'خنثی')
        
        # اضافه کردن سیگنال روند
        trend_emoji = "➡️" if trend == "خنثی" else "📈" if trend == "صعودی" else "📉"
        
        additional_info = f"\n• روند: {trend} {trend_emoji}" if language == "fa" else f"\n• Trend: {trend} {trend_emoji}"
        
        return template.format(
            symbol=symbol,
            rsi=rsi,
            macd=macd,
            support=support,
            resistance=resistance
        ) + additional_info
    
    def _format_fear_greed(self, data: dict, language: str) -> str:
        """فرمت‌بندی شاخص ترس و طمع"""
        template = self.response_templates[language]["fear_greed"]
        
        index = data.get('value', 0)
        status = data.get('status', 'خنثی')
        
        # انتخاب ایموجی بر اساس مقدار شاخص
        if index <= 25:
            emoji = "😱"  # ترس شدید
        elif index <= 45:
            emoji = "😨"  # ترس
        elif index <= 55:
            emoji = "😐"  # خنثی
        elif index <= 75:
            emoji = "😊"  # طمع
        else:
            emoji = "🤩"  # طمع شدید
        
        return template.format(index=index, status=status) + f" {emoji}"
    
    def _format_market_summary(self, data: dict, language: str) -> str:
        """فرمت‌بندی خلاصه بازار"""
        template = self.response_templates[language]["market_summary"]
        
        total_volume = data.get('total_volume', 0)
        gainers = data.get('gainers', 0)
        losers = data.get('losers', 0)
        market_cap = data.get('market_cap', 0)
        
        # اضافه کردن اطلاعات بازار
        additional_info = ""
        if language == "fa":
            additional_info = f"\n• ارزش بازار: ${market_cap:,.0f}"
        else:
            additional_info = f"\n• Market Cap: ${market_cap:,.0f}"
        
        return template.format(
            total_volume=total_volume,
            gainers=gainers,
            losers=losers
        ) + additional_info
    
    def _format_ai_analysis(self, data: dict, language: str) -> str:
        """فرمت‌بندی تحلیل هوش مصنوعی"""
        template = self.response_templates[language]["ai_analysis"]
        
        analysis = data.get('analysis', 'تحصیلی در دسترس نیست')
        confidence = data.get('confidence', 0)
        sentiment = data.get('sentiment', 'خنثی')
        
        # اضافه کردن سطح اطمینان
        confidence_text = f"\n🎯 اطمینان: {confidence:.1%}" if language == "fa" else f"\n🎯 Confidence: {confidence:.1%}"
        
        return template.format(analysis=analysis) + confidence_text
    
    def _format_general_response(self, data: dict, language: str, intent: str = None) -> str:
        """فرمت‌بندی پاسخ عمومی"""
        message = data.get('message', '')
        
        if message:
            if language == "fa":
                return f"✅ {message}"
            else:
                return f"✅ {message}"
        
        # پاسخ پیش‌فرض بر اساس intent
        if intent:
            if language == "fa":
                return f"🤖 درخواست '{intent}' با موفقیت پردازش شد"
            else:
                return f"🤖 Request '{intent}' processed successfully"
        
        return "🤖 پردازش انجام شد" if language == "fa" else "🤖 Processing completed"
    
    def format_typing_indicator(self, language: str = "fa") -> str:
        """ایجاد نشانگر تایپ"""
        if language == "fa":
            return "⏳ در حال تحلیل و پردازش..."
        else:
            return "⏳ Analyzing and processing..."
    
    def format_welcome_message(self, language: str = "fa") -> str:
        """پیام خوشامدگویی"""
        if language == "fa":
            return """🤖 سلام! من دستیار هوشمند VortexAI هستم. 

می‌تونم در زمینه‌های زیر کمکتون کنم:
• 📊 قیمت و تحلیل ارزهای دیجیتال
• 🖥️ وضعیت سیستم و سرور
• 📰 اخبار و تحولات بازار
• 📈 تحلیل تکنیکال و شاخص‌ها
• 🤖 تحلیل هوش مصنوعی داده‌ها

چه سوالی دارید؟"""
        else:
            return """🤖 Hello! I'm VortexAI Smart Assistant.

I can help you with:
• 📊 Cryptocurrency prices and analysis
• 🖥️ System and server status
• 📰 Market news and updates
• 📈 Technical analysis and indicators
• 🤖 AI-powered data analysis

What would you like to know?"""
