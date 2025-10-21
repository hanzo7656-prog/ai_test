# main_ai.py
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys

# اضافه کردن مسیر models به sys.path
sys.path.append(os.path.dirname(__file__))

warnings.filterwarnings('ignore')

# ایمپورت کلاینت و ماژول‌های ما
from api_client import CoinStatsAPIClient
from data_processor import DataProcessor
from risk_manager import RiskManager
import config
import constants

class CryptoAIAnalyst:
    """
    هوش مصنوعی تحلیلگر کامل بازار کریپتو
    """
    
    def __init__(self):
        self.client = CoinStatsAPIClient()
        self.data_processor = DataProcessor()
        self.risk_manager = RiskManager()
        self.market_data = {}
        self.analysis_results = {}
        
        # ایجاد خودکار پوشه‌ها
        self._create_github_directories()
        
        print("🚀 هوش مصنوعی تحلیلگر کریپتو راه‌اندازی شد")
    
    def _create_github_directories(self):
        """ایجاد پوشه‌های مورد نیاز در GitHub"""
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
                print(f"✅ پوشه ایجاد شد: {directory}")
            except Exception as e:
                print(f"⚠️ خطا در ایجاد {directory}: {e}")
        
        # ایجاد فایل realtime_prices.json
        realtime_file = 'shared/realtime_prices.json'
        if not os.path.exists(realtime_file):
            try:
                with open(realtime_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "timestamp": 0, 
                        "realtime_data": {}
                    }, f, indent=2, ensure_ascii=False)
                print(f"✅ فایل ایجاد شد: {realtime_file}")
            except Exception as e:
                print(f"⚠️ خطا در ایجاد فایل: {e}")
    
    def load_market_data(self):
        """بارگذاری کامل داده‌های بازار"""
        print("🔄 در حال بارگذاری داده‌های بازار...")
        
        try:
            # 1. داده‌های اصلی بازار
            self.market_data["coins"] = self.client.get_coins_list(limit=100)
            self.market_data["realtime"] = self.client.get_realtime_data()
            
            # 2. داده‌های تحلیلی
            self.market_data["analytics"] = {
                "fear_greed": self.client.get_fear_greed_index(),
                "btc_dominance": self.client.get_btc_dominance("all"),
                "rainbow_btc": self.client.get_rainbow_chart("bitcoin"),
                "rainbow_eth": self.client.get_rainbow_chart("ethereum")
            }
            
            # 3. اخبار و احساسات
            self.market_data["news"] = {
                "trending": self.client.get_news_by_type("trending"),
                "bullish": self.client.get_news_by_type("bullish"),
                "bearish": self.client.get_news_by_type("bearish")
            }
            
            print("✅ داده‌های بازار با موفقیت بارگذاری شد")
            return True
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری داده‌ها: {e}")
            return False
    
    def technical_analysis(self, coin_data):
        """تحلیل تکنیکال پیشرفته"""
        if not coin_data or len(coin_data) < 20:
            return {"error": "داده ناکافی برای تحلیل تکنیکال"}
        
        # پردازش داده‌ها
        df = self.data_processor.process_chart_data(coin_data)
        df = self.data_processor.calculate_technical_indicators(df)
        
        if df.empty:
            return {"error": "داده ناکافی پس از پاک‌سازی"}
        
        # محاسبه سیگنال‌ها
        current_price = df['price'].iloc[-1] if 'price' in df.columns else 0
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else 0
        sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else 0
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        
        # سیگنال‌های معاملاتی
        signals = {
            "current_price": current_price,
            "trend": "صعودی" if sma_20 > sma_50 else "نزولی",
            "rsi": round(rsi, 2),
            "rsi_signal": "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی",
            "momentum": "قوی" if abs(current_price - sma_20) / sma_20 > 0.05 else "ضعیف",
            "support_level": round(df['price'].min(), 2),
            "resistance_level": round(df['price'].max(), 2),
            "volatility": round(df['price'].std() / df['price'].mean() * 100, 2)
        }
        
        return signals
    
    def sentiment_analysis(self, news_data):
        """تحلیل احساسات بازار"""
        if not news_data:
            return {"sentiment": "خنثی", "confidence": 0.5, "score": 50}
        
        positive_keywords = ['صعود', 'رشد', 'سود', 'خرید', 'بولیش', 'مثبت', 'افزایش', 'قوی']
        negative_keywords = ['نزول', 'سقوط', 'ضرر', 'فروش', 'بیریش', 'منفی', 'کاهش', 'ضعیف']
        
        positive_count = 0
        negative_count = 0
        total_articles = 0
        
        for news_type, articles in news_data.items():
            if isinstance(articles, list):
                for article in articles[:10]:  # 10 خبر اول هر دسته
                    if isinstance(article, dict):
                        title = article.get('title', '').lower()
                        description = article.get('description', '').lower()
                        
                        text = f"{title} {description}"
                        positive_count += sum(1 for word in positive_keywords if word in text)
                        negative_count += sum(1 for word in negative_keywords if word in text)
                        total_articles += 1
        
        if total_articles == 0:
            return {"sentiment": "خنثی", "confidence": 0.5, "score": 50}
        
        sentiment_score = (positive_count / (positive_count + negative_count)) * 100 if (positive_count + negative_count) > 0 else 50
        
        if sentiment_score > 65:
            sentiment = "مثبت"
        elif sentiment_score < 35:
            sentiment = "منفی"
        else:
            sentiment = "خنثی"
        
        confidence = min(abs(sentiment_score - 50) / 50, 1.0)
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "score": round(sentiment_score, 2),
            "positive_news": positive_count,
            "negative_news": negative_count,
            "total_articles": total_articles
        }
    
    def market_health_analysis(self):
        """تحلیل سلامت کلی بازار"""
        fear_greed = self.market_data.get("analytics", {}).get("fear_greed", {})
        btc_dominance = self.market_data.get("analytics", {}).get("btc_dominance", {})
        coins_data = self.market_data.get("coins", {}).get("result", [])
        
        health_score = 0.5  # نمره پایه
        
        # تحلیل شاخص ترس و طمع
        if fear_greed and 'value' in fear_greed:
            fg_value = fear_greed['value']
            if 25 <= fg_value <= 75:
                health_score += 0.2  # بازار سالم
            elif fg_value < 25:
                health_score += 0.1  # ترس زیاد - ممکن است فرصت خرید باشد
            else:
                health_score -= 0.1  # طمع زیاد - خطرناک
        
        # تحلیل تعداد کوین‌های فعال
        if coins_data and len(coins_data) > 50:
            health_score += 0.1  # تنوع خوب
        
        # تحلیل حجم معاملات
        if coins_data:
            total_volume = sum(coin.get('volume', 0) for coin in coins_data[:10])
            if total_volume > 1000000000:  # حجم بالای ۱ میلیارد
                health_score += 0.1
        
        # نرمالایز کردن امتیاز بین ۰ و ۱
        health_score = max(0, min(health_score, 1))
        
        # تعیین وضعیت
        if health_score > 0.7:
            status = "بسیار سالم"
            color = "🟢"
        elif health_score > 0.5:
            status = "سالم"
            color = "🟡"
        elif health_score > 0.3:
            status = "نیازمند احتیاط"
            color = "🟠"
        else:
            status = "پرریسک"
            color = "🔴"
        
        return {
            "health_score": round(health_score, 2),
            "status": status,
            "color": color,
            "fear_greed_index": fear_greed.get('value', 'نامشخص'),
            "active_coins": len(coins_data) if coins_data else 0,
            "market_cap_diversity": "خوب" if len(coins_data) > 50 else "متوسط"
        }
    
    def generate_trading_strategy(self, coin_symbol):
        """تولید استراتژی معاملاتی برای یک کوین"""
        print(f"🎯 تولید استراتژی برای {coin_symbol}...")
        
        # دریافت داده‌های تاریخی
        chart_data = self.client.get_coin_chart(coin_symbol.lower(), "1m")  # یک ماه اخیر
        
        if not chart_data:
            return {"error": "داده تاریخی در دسترس نیست"}
        
        # تحلیل تکنیکال
        tech_analysis = self.technical_analysis(chart_data)
        
        # تحلیل احساسات
        sentiment = self.sentiment_analysis(self.market_data.get("news", {}))
        
        # تحلیل سلامت بازار
        market_health = self.market_health_analysis()
        
        # قیمت لحظه‌ای
        live_price = self.client.get_live_price(coin_symbol + "USDT")
        
        # تولید استراتژی
        strategy = {
            "coin": coin_symbol,
            "timestamp": datetime.now().isoformat(),
            "live_price": live_price,
            "technical_analysis": tech_analysis,
            "sentiment_analysis": sentiment,
            "market_health": market_health,
            "recommendation": self._generate_recommendation(tech_analysis, sentiment, market_health),
            "risk_level": self._calculate_risk_level(tech_analysis, sentiment, market_health),
            "position_sizing": self._calculate_position_size(tech_analysis, market_health),
            "entry_points": self._calculate_entry_points(tech_analysis, live_price),
            "exit_strategy": self._generate_exit_strategy(tech_analysis, live_price)
        }
        
        return strategy
    
    def _generate_recommendation(self, tech_analysis, sentiment, market_health):
        """تولید توصیه معاملاتی"""
        if "error" in tech_analysis:
            return "داده ناکافی - عدم توصیه"
        
        tech_trend = tech_analysis.get("trend", "خنثی")
        rsi_signal = tech_analysis.get("rsi_signal", "خنثی")
        market_sentiment = sentiment.get("sentiment", "خنثی")
        health_status = market_health.get("status", "سالم")
        
        # منطق پیشرفته تصمیم‌گیری
        conditions = []
        
        if tech_trend == "صعودی":
            conditions.append("روند صعودی")
        if market_sentiment == "مثبت":
            conditions.append("احساسات مثبت")
        if health_status in ["سالم", "بسیار سالم"]:
            conditions.append("بازار سالم")
        if rsi_signal == "اشباع فروش":
            conditions.append("اشباع فروش")
        if rsi_signal == "اشباع خرید":
            conditions.append("اشباع خرید")
        
        # تولید توصیه بر اساس شرایط
        if len(conditions) >= 3 and "اشباع خرید" not in conditions:
            return "خرید قوی 📈"
        elif "اشباع خرید" in conditions:
            return "احتیاط در خرید ⚠️"
        elif "اشباع فروش" in conditions and tech_trend == "صعودی":
            return "فرصت خرید خوب ✅"
        elif tech_trend == "نزولی" and market_sentiment == "منفی":
            return "فروش یا انتظار 📉"
        else:
            return "انتظار برای سیگنال clearer 🔄"
    
    def _calculate_risk_level(self, tech_analysis, sentiment, market_health):
        """محاسبه سطح ریسک"""
        risk_score = 0.5
        
        if "error" not in tech_analysis:
            # ریسک بر اساس نوسان
            volatility = tech_analysis.get("volatility", 0)
            if volatility > 10:  # نوسان بالا
                risk_score += 0.3
            elif volatility > 5:
                risk_score += 0.15
            
            # ریسک بر اساس RSI
            if tech_analysis.get("rsi_signal") in ["اشباع خرید", "اشباع فروش"]:
                risk_score += 0.1
        
        # ریسک بر اساس احساسات
        sentiment_confidence = sentiment.get("confidence", 0.5)
        risk_score += (1 - sentiment_confidence) * 0.2
        
        # ریسک بر اساس سلامت بازار
        health_score = market_health.get("health_score", 0.5)
        risk_score += (1 - health_score) * 0.2
        
        # تعیین سطح ریسک
        if risk_score > 0.7:
            return "بالا 🔴"
        elif risk_score > 0.4:
            return "متوسط 🟡"
        else:
            return "پایین 🟢"
    
    def _calculate_position_size(self, tech_analysis, market_health):
        """محاسبه سایز پوزیشن"""
        base_size = 0.1  # 10% پایه
        
        if "error" in tech_analysis:
            return f"{base_size * 100}% (پایه - داده ناکافی)"
        
        # تنظیم بر اساس تحلیل تکنیکال
        if tech_analysis.get("trend") == "صعودی":
            base_size += 0.1
        if tech_analysis.get("rsi_signal") == "اشباع فروش":
            base_size += 0.05
        
        # تنظیم بر اساس سلامت بازار
        health_status = market_health.get("status", "سالم")
        if health_status == "پرریسک":
            base_size *= 0.5  # نصف کردن پوزیشن
        elif health_status == "بسیار سالم":
            base_size *= 1.2  # افزایش 20%
        
        return f"{min(base_size * 100, 30)}% از سرمایه"  # حداکثر 30%
    
    def _calculate_entry_points(self, tech_analysis, live_price):
        """محاسبه نقاط ورود"""
        if "error" in tech_analysis or not live_price:
            return ["داده ناکافی"]
        
        support = tech_analysis.get("support_level", 0)
        resistance = tech_analysis.get("resistance_level", 0)
        current_price = tech_analysis.get("current_price", live_price)
        
        entry_points = []
        
        # نقطه ورود محافظه‌کارانه
        conservative_entry = support * 1.02  # 2% بالاتر از ساپورت
        if conservative_entry < current_price:
            entry_points.append(f"ورود محافظه‌کارانه: ${conservative_entry:,.2f}")
        
        # نقطه ورود تهاجمی
        aggressive_entry = current_price * 0.98  # 2% زیر قیمت فعلی
        entry_points.append(f"ورود تهاجمی: ${aggressive_entry:,.2f}")
        
        return entry_points
    
    def _generate_exit_strategy(self, tech_analysis, live_price):
        """تولید استراتژی خروج"""
        if "error" in tech_analysis or not live_price:
            return {"take_profit": "داده ناکافی", "stop_loss": "داده ناکافی"}
        
        current_price = tech_analysis.get("current_price", live_price)
        resistance = tech_analysis.get("resistance_level", current_price * 1.1)
        support = tech_analysis.get("support_level", current_price * 0.9)
        
        return {
            "take_profit": f"${resistance:,.2f} ({((resistance - current_price) / current_price * 100):.1f}%)",
            "stop_loss": f"${support:,.2f} ({((current_price - support) / current_price * 100):.1f}%)",
            "risk_reward_ratio": f"{((resistance - current_price) / (current_price - support)):.2f}:1"
        }
    
    def comprehensive_analysis(self, top_coins=5):
        """تحلیل جامع بازار"""
        print("🧠 شروع تحلیل جامع بازار...")
        
        if not self.market_data:
            success = self.load_market_data()
            if not success:
                return {"error": "خطا در بارگذاری داده‌های بازار"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "market_health": self.market_health_analysis(),
            "sentiment_analysis": self.sentiment_analysis(self.market_data.get("news", {})),
            "top_coins_analysis": [],
            "overall_recommendation": "",
            "risk_assessment": {},
            "api_status": self.client.get_api_status()
        }
        
        # تحلیل کوین‌های برتر
        coins_data = self.market_data.get("coins", {}).get("result", [])
        analyzed_coins = 0
        
        for coin in coins_data[:top_coins]:
            symbol = coin.get('symbol')
            if symbol and analyzed_coins < top_coins:
                strategy = self.generate_trading_strategy(symbol)
                if "error" not in strategy:
                    results["top_coins_analysis"].append(strategy)
                    analyzed_coins += 1
        
        # ارزیابی ریسک کلی
        results["risk_assessment"] = {
            "market_risk": results["market_health"]["status"],
            "sentiment_risk": results["sentiment_analysis"]["sentiment"],
            "overall_risk": "بالا" if (results["market_health"]["status"] == "پرریسک" or 
                                     results["sentiment_analysis"]["sentiment"] == "منفی") else "متوسط"
        }
        
        # توصیه کلی
        health_status = results["market_health"]["status"]
        sentiment = results["sentiment_analysis"]["sentiment"]
        
        if health_status in ["بسیار سالم", "سالم"] and sentiment == "مثبت":
            results["overall_recommendation"] = "شرایط مطلوب برای سرمایه‌گذاری 🎯"
        elif health_status == "پرریسک" or sentiment == "منفی":
            results["overall_recommendation"] = "احتیاط - کاهش حجم معاملات ⚠️"
        else:
            results["overall_recommendation"] = "انتظار برای شرایط بهتر بهتر 🔄"
        
        print("✅ تحلیل جامع تکمیل شد")
        
        # ذخیره نتایج
        self._save_analysis_results(results)
        
        return results
    
    def _save_analysis_results(self, results):
        """ذخیره نتایج تحلیل"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/analysis/analysis_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 نتایج در {filename} ذخیره شد")
            
        except Exception as e:
            print(f"⚠️ خطا در ذخیره نتایج: {e}")
    
    def print_analysis_summary(self, analysis):
        """چاپ خلاصه تحلیل"""
        print("\n" + "="*60)
        print("📊 خلاصه تحلیل هوش مصنوعی بازار کریپتو")
        print("="*60)
        
        # سلامت بازار
        health = analysis['market_health']
        print(f"\n🏥 سلامت بازار: {health['color']} {health['status']} (امتیاز: {health['health_score']})")
        print(f"😨 شاخص ترس و طمع: {health['fear_greed_index']}")
        
        # احساسات
        sentiment = analysis['sentiment_analysis']
        print(f"😊 احساسات بازار: {sentiment['sentiment']} (امتیاز: {sentiment['score']})")
        print(f"📰 اخبار مثبت/منفی: {sentiment['positive_news']}/{sentiment['negative_news']}")
        
        # توصیه کلی
        print(f"\n💡 توصیه کلی: {analysis['overall_recommendation']}")
        
        # ریسک
        risk = analysis['risk_assessment']
        print(f"⚠️ ارزیابی ریسک: {risk['overall_risk']}")
        
        # تحلیل کوین‌ها
        print(f"\n🎯 استراتژی‌های کوین‌های برتر:")
        for coin_analysis in analysis['top_coins_analysis']:
            coin = coin_analysis['coin']
            recommendation = coin_analysis['recommendation']
            risk_level = coin_analysis['risk_level']
            live_price = coin_analysis.get('live_price', 'نامشخص')
            
            print(f"  • {coin}: {recommendation}")
            print(f"    💰 قیمت: {live_price} | 🎯 ریسک: {risk_level}")
            
            # نمایش نقاط ورود
            entry_points = coin_analysis.get('entry_points', [])
            if entry_points and len(entry_points) > 0:
                print(f"    📍 {entry_points[0]}")
        
        print(f"\n⏰ آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)


# اجرای اصلی
if __name__ == "__main__":
    print("🚀 راه‌اندازی هوش مصنوعی تحلیلگر کریپتو...")
    
    try:
        # ایجاد تحلیلگر
        ai_analyst = CryptoAIAnalyst()
        
        # تحلیل جامع
        analysis = ai_analyst.comprehensive_analysis(top_coins=5)
        
        if "error" not in analysis:
            # نمایش نتایج
            ai_analyst.print_analysis_summary(analysis)
        else:
            print(f"❌ خطا: {analysis['error']}")
            
    except Exception as e:
        print(f"❌ خطای غیرمنتظره: {e}")
        import traceback
        traceback.print_exc()
