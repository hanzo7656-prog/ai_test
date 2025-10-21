# main_ai.py
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys
import psutil
import logging
import traceback

# اضافه کردن مسیر فعلی به sys.path
sys.path.append(os.path.dirname(__file__))

warnings.filterwarnings('ignore')

# تنظیمات لاگ برای Render
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ایمپورت کلاینت و ماژول‌های ما
try:
    from api_client import CoinStatsAPIClient
    from data_processor import DataProcessor
    from risk_manager import RiskManager
except ImportError as e:
    logger.error(f"خطا در ایمپورت ماژول‌ها: {e}")

class CryptoAIAnalyst:
    """هوش مصنوعی تحلیلگر کامل بازار کریپتو"""
    
    def __init__(self):
        self.client = CoinStatsAPIClient()
        self.data_processor = DataProcessor()
        self.risk_manager = RiskManager()
        self.market_data = {}
        self.analysis_results = {}
        self.performance_metrics = {}
        
        # ایجاد خودکار پوشه‌ها (با مدیریت خطا برای Render)
        self._create_directories_safe()
        
        # تاریخچه تحلیل‌ها
        self.analysis_history = []
        
        logger.info("🚀 هوش مصنوعی تحلیلگر کریپتو راه‌اندازی شد")
    
    def _create_directories_safe(self):
        """ایجاد پوشه‌های مورد نیاز با مدیریت خطا"""
        directories = [
            'shared',
            'data/historical',
            'data/analysis',
            'data/models', 
            'data/snapshots'
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✅ پوشه ایجاد شد: {directory}")
            except Exception as e:
                logger.debug(f"⚠️ خطا در ایجاد {directory}: {e}")
    
    def load_market_data(self, force_refresh: bool = False) -> bool:
        """بارگذاری کامل داده‌های بازار"""
        logger.info("🔄 در حال بارگذاری داده‌های بازار...")
        
        try:
            # 1. داده‌های اصلی بازار
            self.market_data["coins"] = self.client.get_coins_list(limit=100, use_local=True)
            
            # 2. داده‌های تحلیلی
            self.market_data["analytics"] = self.client.get_analytical_data(use_local=True)
            
            # 3. اخبار و احساسات
            self.market_data["news"] = {
                "trending": self.client.get_news_by_type("trending"),
                "bullish": self.client.get_news_by_type("bullish"),
                "bearish": self.client.get_news_by_type("bearish")
            }
            
            logger.info("✅ داده‌های بازار با موفقیت بارگذاری شد")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری داده‌ها: {e}")
            return False
    
    def technical_analysis(self, coin_symbol: str, period: str = "1m") -> dict:
        """تحلیل تکنیکال پیشرفته برای یک کوین"""
        logger.info(f"📈 تحلیل تکنیکال برای {coin_symbol} ({period})")
        
        try:
            chart_data = self.client.get_coin_chart(coin_symbol.lower(), period, use_local=True)
            
            if not chart_data:
                return {"error": "داده تاریخی در دسترس نیست"}
            
            processed_data = self.data_processor.process_chart_data(chart_data)
            
            if processed_data.empty:
                return {"error": "داده‌های پردازش شده خالی است"}
            
            with_indicators = self.data_processor.calculate_technical_indicators(processed_data)
            signals = self.data_processor.generate_trading_signals(with_indicators)
            summary = self.data_processor.get_technical_summary(with_indicators)
            
            analysis_result = {
                "coin": coin_symbol,
                "period": period,
                "data_points": len(processed_data),
                "technical_analysis": signals,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ تحلیل تکنیکال برای {coin_symbol} تکمیل شد")
            return analysis_result
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل تکنیکال {coin_symbol}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def sentiment_analysis(self) -> dict:
        """تحلیل احساسات بازار از اخبار"""
        logger.info("😊 تحلیل احساسات بازار...")
        
        try:
            news_data = self.market_data.get("news", {})
            
            if not news_data:
                return {"sentiment": "خنثی", "confidence": 0.5, "score": 50}
            
            positive_keywords = ['صعود', 'رشد', 'سود', 'خرید', 'بولیش', 'مثبت']
            negative_keywords = ['نزول', 'سقوط', 'ضرر', 'فروش', 'بیریش', 'منفی']
            
            positive_count = 0
            negative_count = 0
            total_articles = 0
            
            for news_type, articles in news_data.items():
                if isinstance(articles, list):
                    for article in articles[:10]:
                        if isinstance(article, dict):
                            title = article.get('title', '').lower()
                            description = article.get('description', '').lower()
                            
                            text = f"{title} {description}"
                            positive_count += sum(1 for word in positive_keywords if word in text)
                            negative_count += sum(1 for word in negative_keywords if word in text)
                            total_articles += 1
            
            if total_articles == 0:
                return {"sentiment": "خنثی", "confidence": 0.5, "score": 50, "total_articles": 0}
            
            total_keywords = positive_count + negative_count
            sentiment_score = (positive_count / total_keywords * 100) if total_keywords > 0 else 50
            
            if sentiment_score > 65:
                sentiment = "مثبت شدید 🟢"
            elif sentiment_score > 55:
                sentiment = "مثبت 🟢"
            elif sentiment_score > 45:
                sentiment = "خنثی 🟡"
            elif sentiment_score > 35:
                sentiment = "منفی 🔴"
            else:
                sentiment = "منفی شدید 🔴"
            
            confidence = min(abs(sentiment_score - 50) / 50, 1.0)
            
            result = {
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "score": round(sentiment_score, 2),
                "positive_news": positive_count,
                "negative_news": negative_count,
                "total_articles": total_articles
            }
            
            logger.info(f"✅ تحلیل احساسات تکمیل شد: {sentiment} (امتیاز: {sentiment_score})")
            return result
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل احساسات: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def market_health_analysis(self) -> dict:
        """تحلیل سلامت کلی بازار"""
        logger.info("🏥 تحلیل سلامت بازار...")
        
        try:
            fear_greed = self.market_data.get("analytics", {}).get("fear_greed", {})
            coins_data = self.market_data.get("coins", {}).get("result", [])
            
            health_score = 0.5
            
            # تحلیل شاخص ترس و طمع
            if fear_greed and 'value' in fear_greed:
                fg_value = fear_greed['value']
                if 25 <= fg_value <= 75:
                    health_score += 0.2
                elif fg_value < 25:
                    health_score += 0.1
                else:
                    health_score -= 0.1
            
            # تحلیل تعداد کوین‌های فعال
            if coins_data and len(coins_data) > 50:
                health_score += 0.1
            
            health_score = max(0, min(health_score, 1))
            
            if health_score > 0.7:
                status = "بسیار سالم 🟢"
            elif health_score > 0.5:
                status = "سالم 🟢"
            elif health_score > 0.3:
                status = "نیازمند احتیاط 🟠"
            else:
                status = "پرریسک 🔴"
            
            result = {
                "health_score": round(health_score, 3),
                "status": status,
                "fear_greed_index": fear_greed.get('value', 'نامشخص'),
                "active_coins": len(coins_data) if coins_data else 0
            }
            
            logger.info(f"✅ تحلیل سلامت بازار تکمیل شد: {status} (امتیاز: {health_score})")
            return result
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل سلامت بازار: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def generate_trading_strategy(self, coin_symbol: str) -> dict:
        """تولید استراتژی معاملاتی برای یک کوین"""
        logger.info(f"🎯 تولید استراتژی برای {coin_symbol}...")
        
        try:
            tech_analysis = self.technical_analysis(coin_symbol, "1m")
            sentiment = self.sentiment_analysis()
            market_health = self.market_health_analysis()
            
            strategy = {
                "coin": coin_symbol,
                "timestamp": datetime.now().isoformat(),
                "technical_analysis": tech_analysis,
                "sentiment_analysis": sentiment,
                "market_health": market_health,
                "recommendation": self._generate_recommendation(tech_analysis, sentiment, market_health),
                "risk_level": self._calculate_risk_level(tech_analysis, sentiment, market_health),
                "position_sizing": "5-10% از سرمایه",  # ساده‌سازی شده
                "timeframe": "میان‌مدت (۱-۴ هفته) 📅"
            }
            
            logger.info(f"✅ استراتژی برای {coin_symbol} تولید شد: {strategy['recommendation']}")
            return strategy
            
        except Exception as e:
            error_msg = f"❌ خطا در تولید استراتژی برای {coin_symbol}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _generate_recommendation(self, tech_analysis: dict, sentiment: dict, market_health: dict) -> str:
        """تولید توصیه معاملاتی"""
        if "error" in tech_analysis or "error" in sentiment or "error" in market_health:
            return "داده ناکافی - عدم توصیه 📊"
        
        tech_data = tech_analysis.get("technical_analysis", {})
        sentiment_score = sentiment.get("score", 50)
        
        signals = []
        
        if "rsi" in tech_data:
            rsi_signal = tech_data["rsi"].get("signal", "خنثی")
            if rsi_signal == "اشباع فروش":
                signals.append("RSI اشباع فروش")
        
        if sentiment_score > 60:
            signals.append("احساسات مثبت")
        
        buy_signals = len([s for s in signals if "اشباع فروش" in s or "مثبت" in s])
        
        if buy_signals >= 2:
            return "خرید قوی 📈"
        elif buy_signals >= 1:
            return "خرید متوسط ✅"
        else:
            return "انتظار برای سیگنال clearer 📊"
    
    def _calculate_risk_level(self, tech_analysis: dict, sentiment: dict, market_health: dict) -> str:
        """محاسبه سطح ریسک"""
        risk_score = 0.5
        
        if "error" not in tech_analysis:
            tech_data = tech_analysis.get("technical_analysis", {})
            if "risk_metrics" in tech_data and "volatility" in tech_data["risk_metrics"]:
                vol_level = tech_data["risk_metrics"]["volatility"].get("level", "متوسط")
                if vol_level == "بالا":
                    risk_score += 0.3
        
        sentiment_confidence = sentiment.get("confidence", 0.5)
        risk_score += (1 - sentiment_confidence) * 0.2
        
        if risk_score > 0.7:
            return "بالا 🔴"
        elif risk_score > 0.4:
            return "متوسط 🟡"
        else:
            return "پایین 🟢"
    
    def comprehensive_analysis(self, top_coins: int = 3) -> dict:
        """تحلیل جامع بازار"""
        logger.info(f"🧠 شروع تحلیل جامع بازار برای {top_coins} کوین برتر...")
        
        start_time = datetime.now()
        
        try:
            if not self.market_data:
                success = self.load_market_data()
                if not success:
                    return {"error": "خطا در بارگذاری داده‌های بازار"}
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "analysis_duration": "",
                "market_health": self.market_health_analysis(),
                "sentiment_analysis": self.sentiment_analysis(),
                "top_coins_analysis": [],
                "overall_recommendation": "",
                "system_status": self.get_system_status()
            }
            
            # تحلیل کوین‌های برتر
            coins_data = self.market_data.get("coins", {}).get("result", [])
            analyzed_coins = 0
            
            for coin in coins_data[:top_coins]:
                symbol = coin.get('symbol')
                name = coin.get('name', 'نامشخص')
                
                if symbol and analyzed_coins < top_coins:
                    logger.info(f"  📊 تحلیل {name} ({symbol})...")
                    strategy = self.generate_trading_strategy(symbol)
                    
                    if "error" not in strategy:
                        strategy["coin_info"] = {
                            "name": name,
                            "rank": coin.get('rank', 0),
                            "market_cap": coin.get('marketCap', 0)
                        }
                        results["top_coins_analysis"].append(strategy)
                        analyzed_coins += 1
            
            # توصیه کلی
            health_status = results["market_health"].get("status", "")
            sentiment = results["sentiment_analysis"].get("sentiment", "")
            
            if "سالم" in health_status and "مثبت" in sentiment:
                results["overall_recommendation"] = "شرایط مطلوب برای سرمایه‌گذاری 🎯"
            elif "پرریسک" in health_status or "منفی" in sentiment:
                results["overall_recommendation"] = "احتیاط - کاهش حجم معاملات ⚠️"
            else:
                results["overall_recommendation"] = "شرایط متوسط - انتخاب‌های محتاطانه 🔄"
            
            # محاسبه مدت زمان تحلیل
            duration = datetime.now() - start_time
            results["analysis_duration"] = f"{duration.total_seconds():.1f} ثانیه"
            
            logger.info(f"✅ تحلیل جامع تکمیل شد (زمان: {duration.total_seconds():.1f} ثانیه)")
            return results
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل جامع: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_system_status(self) -> dict:
        """بررسی وضعیت سیستم و حافظه"""
        try:
            return {
                "memory_usage": "N/A",  # ساده‌سازی برای Render
                "cpu_usage": "N/A",
                "python_version": sys.version,
                "running_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "operational"
            }
        except Exception as e:
            return {"error": f"خطا در بررسی وضعیت سیستم: {e}"}

# تابع اصلی برای اجرای مستقل
def main():
    """تابع اصلی برای اجرای مستقیم"""
    print("🚀 راه‌اندازی هوش مصنوعی تحلیلگر کریپتو...")
    
    try:
        ai_analyst = CryptoAIAnalyst()
        analysis = ai_analyst.comprehensive_analysis(top_coins=3)
        
        if "error" not in analysis:
            print(f"\n📊 تحلیل کامل شد!")
            print(f"🏥 سلامت بازار: {analysis['market_health']['status']}")
            print(f"😊 احساسات: {analysis['sentiment_analysis']['sentiment']}")
            print(f"💡 توصیه کلی: {analysis['overall_recommendation']}")
            
            for coin_analysis in analysis['top_coins_analysis']:
                coin = coin_analysis['coin']
                recommendation = coin_analysis['recommendation']
                risk_level = coin_analysis['risk_level']
                print(f"  • {coin}: {recommendation} (ریسک: {risk_level})")
        else:
            print(f"❌ خطا: {analysis['error']}")
            
    except Exception as e:
        print(f"❌ خطای غیرمنتظره: {e}")

if __name__ == "__main__":
    main()
