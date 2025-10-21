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
from flask import Flask, jsonify
import traceback

# اضافه کردن مسیر فعلی به sys.path
sys.path.append(os.path.dirname(__file__))

warnings.filterwarnings('ignore')

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/analysis/ai_analyst.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ایمپورت کلاینت و ماژول‌های ما
try:
    from api_client import CoinStatsAPIClient
    from data_processor import DataProcessor
    from risk_manager import RiskManager
    import config
    import constants
except ImportError as e:
    logger.error(f"خطا در ایمپورت ماژول‌ها: {e}")
    sys.exit(1)

class CryptoAIAnalyst:
    """
    هوش مصنوعی تحلیلگر کامل بازار کریپتو با قابلیت‌های پیشرفته
    """
    
    def __init__(self):
        self.client = CoinStatsAPIClient()
        self.data_processor = DataProcessor()
        self.risk_manager = RiskManager()
        self.market_data = {}
        self.analysis_results = {}
        self.performance_metrics = {}
        
        # ایجاد خودکار پوشه‌ها
        self._create_directories()
        
        # تاریخچه تحلیل‌ها
        self.analysis_history = []
        
        logger.info("🚀 هوش مصنوعی تحلیلگر کریپتو راه‌اندازی شد")
    
    def _create_directories(self):
        """ایجاد پوشه‌های مورد نیاز"""
        directories = [
            'shared',
            'data/historical',
            'data/analysis',
            'data/models', 
            'data/snapshots',
            'data/logs'
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✅ پوشه ایجاد شد: {directory}")
            except Exception as e:
                logger.warning(f"⚠️ خطا در ایجاد {directory}: {e}")
        
        # ایجاد فایل realtime_prices.json
        realtime_file = 'shared/realtime_prices.json'
        if not os.path.exists(realtime_file):
            try:
                with open(realtime_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "timestamp": 0, 
                        "realtime_data": {},
                        "last_updated": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ فایل ایجاد شد: {realtime_file}")
            except Exception as e:
                logger.error(f"❌ خطا در ایجاد فایل: {e}")
    
    def load_market_data(self, force_refresh: bool = False) -> bool:
        """بارگذاری کامل داده‌های بازار"""
        logger.info("🔄 در حال بارگذاری داده‌های بازار...")
        
        try:
            # 1. داده‌های اصلی بازار
            self.market_data["coins"] = self.client.get_coins_list(
                limit=150, 
                include_risk_score=True
            )
            
            # داده‌های real-time
            self.market_data["realtime"] = self.client.get_realtime_data()
            
            # 2. داده‌های تحلیلی
            self.market_data["analytics"] = {
                "fear_greed": self.client.get_fear_greed_index(),
                "fear_greed_chart": self.client.get_fear_greed_chart(),
                "btc_dominance": self.client.get_btc_dominance("all"),
                "rainbow_btc": self.client.get_rainbow_chart("bitcoin"),
                "rainbow_eth": self.client.get_rainbow_chart("ethereum")
            }
            
            # 3. اخبار و احساسات
            self.market_data["news"] = {
                "trending": self.client.get_news_by_type("trending"),
                "latest": self.client.get_news_by_type("latest"),
                "bullish": self.client.get_news_by_type("bullish"),
                "bearish": self.client.get_news_by_type("bearish")
            }
            
            # 4. داده‌های بازار
            self.market_data["market_info"] = {
                "exchanges": self.client.get_exchanges(),
                "markets": self.client.get_markets(),
                "fiats": self.client.get_fiats()
            }
            
            # ذخیره snapshot
            self._save_market_snapshot()
            
            logger.info("✅ داده‌های بازار با موفقیت بارگذاری شد")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری داده‌ها: {e}")
            return False
    
    def _save_market_snapshot(self):
        """ذخیره عکس‌العمل از بازار"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_file = f"data/snapshots/market_snapshot_{timestamp}.json"
            
            snapshot_data = {
                "timestamp": datetime.now().isoformat(),
                "market_data": {
                    "total_coins": len(self.market_data.get("coins", {}).get("result", [])),
                    "fear_greed": self.market_data.get("analytics", {}).get("fear_greed", {}),
                    "btc_dominance": self.market_data.get("analytics", {}).get("btc_dominance", {})
                },
                "system_status": self.get_system_status()
            }
            
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"💾 عکس‌العمل بازار در {snapshot_file} ذخیره شد")
            
        except Exception as e:
            logger.error(f"⚠️ خطا در ذخیره عکس‌العمل: {e}")
    
    def technical_analysis(self, coin_symbol: str, period: str = "1m") -> Dict:
        """تحلیل تکنیکال پیشرفته برای یک کوین"""
        logger.info(f"📈 تحلیل تکنیکال برای {coin_symbol} ({period})")
        
        try:
            # دریافت داده‌های تاریخی
            chart_data = self.client.get_coin_chart(coin_symbol.lower(), period)
            
            if not chart_data:
                return {"error": "داده تاریخی در دسترس نیست"}
            
            # پردازش داده‌ها
            processed_data = self.data_processor.process_chart_data(chart_data)
            
            if processed_data.empty:
                return {"error": "داده‌های پردازش شده خالی است"}
            
            # محاسبه اندیکاتورها
            with_indicators = self.data_processor.calculate_technical_indicators(processed_data)
            
            # تولید سیگنال‌ها
            signals = self.data_processor.generate_trading_signals(with_indicators)
            
            # خلاصه تحلیل
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
    
    def sentiment_analysis(self) -> Dict:
        """تحلیل احساسات بازار از اخبار"""
        logger.info("😊 تحلیل احساسات بازار...")
        
        try:
            news_data = self.market_data.get("news", {})
            
            if not news_data:
                return {"sentiment": "خنثی", "confidence": 0.5, "score": 50}
            
            positive_keywords = [
                'صعود', 'رشد', 'سود', 'خرید', 'بولیش', 'مثبت', 'افزایش', 
                'قوی', 'موفق', 'سرمایه‌گذاری', 'فرصت', 'بهبود'
            ]
            
            negative_keywords = [
                'نزول', 'سقوط', 'ضرر', 'فروش', 'بیریش', 'منفی', 'کاهش',
                'ضعیف', 'شکست', 'هشدار', 'ریسک', 'حباب'
            ]
            
            positive_count = 0
            negative_count = 0
            total_articles = 0
            analyzed_text = ""
            
            for news_type, articles in news_data.items():
                if isinstance(articles, list):
                    for article in articles[:15]:  # 15 خبر اول هر دسته
                        if isinstance(article, dict):
                            title = article.get('title', '').lower()
                            description = article.get('description', '').lower()
                            
                            text = f"{title} {description}"
                            analyzed_text += text + " "
                            
                            positive_count += sum(1 for word in positive_keywords if word in text)
                            negative_count += sum(1 for word in negative_keywords if word in text)
                            total_articles += 1
            
            if total_articles == 0:
                return {
                    "sentiment": "خنثی", 
                    "confidence": 0.5, 
                    "score": 50,
                    "total_articles": 0
                }
            
            total_keywords = positive_count + negative_count
            sentiment_score = (positive_count / total_keywords * 100) if total_keywords > 0 else 50
            
            # تعیین احساسات
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
            
            # تحلیل کلمات کلیدی
            keyword_analysis = {
                "top_positive": [],
                "top_negative": []
            }
            
            for keyword in positive_keywords:
                if keyword in analyzed_text:
                    keyword_analysis["top_positive"].append(keyword)
            
            for keyword in negative_keywords:
                if keyword in analyzed_text:
                    keyword_analysis["top_negative"].append(keyword)
            
            result = {
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "score": round(sentiment_score, 2),
                "positive_news": positive_count,
                "negative_news": negative_count,
                "total_articles": total_articles,
                "keyword_analysis": keyword_analysis,
                "news_sources": list(news_data.keys())
            }
            
            logger.info(f"✅ تحلیل احساسات تکمیل شد: {sentiment} (امتیاز: {sentiment_score})")
            return result
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل احساسات: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def market_health_analysis(self) -> Dict:
        """تحلیل سلامت کلی بازار"""
        logger.info("🏥 تحلیل سلامت بازار...")
        
        try:
            fear_greed = self.market_data.get("analytics", {}).get("fear_greed", {})
            btc_dominance = self.market_data.get("analytics", {}).get("btc_dominance", {})
            coins_data = self.market_data.get("coins", {}).get("result", [])
            realtime_data = self.market_data.get("realtime", {})
            
            health_score = 0.5  # نمره پایه
            
            # 1. تحلیل شاخص ترس و طمع (40% وزن)
            if fear_greed and 'value' in fear_greed:
                fg_value = fear_greed['value']
                if 25 <= fg_value <= 75:
                    health_score += 0.2  # بازار سالم
                elif fg_value < 25:
                    health_score += 0.1  # ترس زیاد - ممکن است فرصت خرید باشد
                else:
                    health_score -= 0.1  # طمع زیاد - خطرناک
            else:
                health_score -= 0.1  # داده نامعتبر
            
            # 2. تحلیل تنوع بازار (20% وزن)
            if coins_data:
                total_coins = len(coins_data)
                if total_coins > 100:
                    health_score += 0.1  # تنوع عالی
                elif total_coins > 50:
                    health_score += 0.05  # تنوع خوب
                
                # تحلیل مارکت‌کپ
                large_cap_count = sum(1 for coin in coins_data if coin.get('marketCap', 0) > 1000000000)
                if large_cap_count > 10:
                    health_score += 0.05
            
            # 3. تحلیل حجم معاملات (20% وزن)
            if coins_data:
                total_volume = sum(coin.get('volume', 0) for coin in coins_data[:20])
                if total_volume > 5000000000:  # حجم بالای ۵ میلیارد
                    health_score += 0.1
                elif total_volume > 1000000000:  # حجم بالای ۱ میلیارد
                    health_score += 0.05
            
            # 4. تحلیل داده‌های real-time (20% وزن)
            if realtime_data and len(realtime_data) > 10:
                health_score += 0.1  # داده‌های لحظه‌ای در دسترس
            
            # نرمالایز کردن امتیاز بین ۰ و ۱
            health_score = max(0, min(health_score, 1))
            
            # تعیین وضعیت
            if health_score > 0.75:
                status = "بسیار سالم 🟢"
                color = "green"
            elif health_score > 0.6:
                status = "سالم 🟢"
                color = "lightgreen"
            elif health_score > 0.45:
                status = "متوسط 🟡"
                color = "yellow"
            elif health_score > 0.3:
                status = "نیازمند احتیاط 🟠"
                color = "orange"
            else:
                status = "پرریسک 🔴"
                color = "red"
            
            result = {
                "health_score": round(health_score, 3),
                "status": status,
                "color": color,
                "fear_greed_index": fear_greed.get('value', 'نامشخص'),
                "fear_greed_label": fear_greed.get('label', 'نامشخص'),
                "active_coins": len(coins_data) if coins_data else 0,
                "market_cap_diversity": "عالی" if len(coins_data) > 100 else "خوب" if len(coins_data) > 50 else "متوسط",
                "volume_health": "قوی" if total_volume > 5000000000 else "متوسط" if total_volume > 1000000000 else "ضعیف",
                "realtime_data_available": len(realtime_data) if realtime_data else 0
            }
            
            logger.info(f"✅ تحلیل سلامت بازار تکمیل شد: {status} (امتیاز: {health_score})")
            return result
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل سلامت بازار: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def generate_trading_strategy(self, coin_symbol: str) -> Dict:
        """تولید استراتژی معاملاتی برای یک کوین"""
        logger.info(f"🎯 تولید استراتژی برای {coin_symbol}...")
        
        try:
            # تحلیل تکنیکال
            tech_analysis = self.technical_analysis(coin_symbol, "1m")
            
            # تحلیل احساسات
            sentiment = self.sentiment_analysis()
            
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
                "exit_strategy": self._generate_exit_strategy(tech_analysis, live_price),
                "timeframe": self._recommend_timeframe(tech_analysis),
                "confidence_score": self._calculate_confidence(tech_analysis, sentiment, market_health)
            }
            
            logger.info(f"✅ استراتژی برای {coin_symbol} تولید شد: {strategy['recommendation']}")
            return strategy
            
        except Exception as e:
            error_msg = f"❌ خطا در تولید استراتژی برای {coin_symbol}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _generate_recommendation(self, tech_analysis: Dict, sentiment: Dict, market_health: Dict) -> str:
        """تولید توصیه معاملاتی"""
        if "error" in tech_analysis or "error" in sentiment or "error" in market_health:
            return "داده ناکافی - عدم توصیه 📊"
        
        tech_data = tech_analysis.get("technical_analysis", {})
        sentiment_score = sentiment.get("score", 50)
        health_status = market_health.get("status", "")
        
        # جمع‌آوری سیگنال‌ها
        signals = []
        
        # سیگنال‌های تکنیکال
        if "rsi" in tech_data:
            rsi_signal = tech_data["rsi"].get("signal", "خنثی")
            if rsi_signal == "اشباع فروش":
                signals.append("RSI اشباع فروش")
            elif rsi_signal == "اشباع خرید":
                signals.append("RSI اشباع خرید")
        
        if "macd" in tech_data:
            macd_trend = tech_data["macd"].get("trend", "خنثی")
            if macd_trend == "صعودی":
                signals.append("MACD صعودی")
        
        # سیگنال‌های احساساتی
        if sentiment_score > 60:
            signals.append("احساسات مثبت")
        elif sentiment_score < 40:
            signals.append("احساسات منفی")
        
        # سیگنال‌های سلامت بازار
        if "سالم" in health_status or "بسیار سالم" in health_status:
            signals.append("بازار سالم")
        
        # منطق تصمیم‌گیری
        buy_signals = sum(1 for s in signals if any(word in s for word in ["اشباع فروش", "صعودی", "مثبت", "سالم"]))
        sell_signals = sum(1 for s in signals if any(word in s for word in ["اشباع خرید", "منفی"]))
        
        if buy_signals >= 3:
            return "خرید قوی 📈"
        elif buy_signals >= 2:
            return "خرید متوسط ✅"
        elif sell_signals >= 3:
            return "فروش قوی 📉"
        elif sell_signals >= 2:
            return "فروش متوسط ⚠️"
        elif buy_signals > sell_signals:
            return "احتیاط در خرید 🔄"
        else:
            return "انتظار برای سیگنال clearer 📊"
    
    def _calculate_risk_level(self, tech_analysis: Dict, sentiment: Dict, market_health: Dict) -> str:
        """محاسبه سطح ریسک"""
        risk_score = 0.5
        
        if "error" not in tech_analysis:
            tech_data = tech_analysis.get("technical_analysis", {})
            
            # ریسک بر اساس نوسان
            if "risk_metrics" in tech_data and "volatility" in tech_data["risk_metrics"]:
                vol_level = tech_data["risk_metrics"]["volatility"].get("level", "متوسط")
                if vol_level == "بالا":
                    risk_score += 0.3
                elif vol_level == "متوسط":
                    risk_score += 0.15
            
            # ریسک بر اساس RSI
            if "rsi" in tech_data:
                rsi_signal = tech_data["rsi"].get("signal", "خنثی")
                if rsi_signal in ["اشباع خرید", "اشباع فروش"]:
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
    
    def _calculate_position_size(self, tech_analysis: Dict, market_health: Dict) -> str:
        """محاسبه سایز پوزیشن"""
        base_size = 0.1  # 10% پایه
        
        if "error" in tech_analysis:
            return f"{base_size * 100}% (پایه - داده ناکافی)"
        
        # تنظیم بر اساس تحلیل تکنیکال
        tech_data = tech_analysis.get("technical_analysis", {})
        if "rsi" in tech_data:
            rsi_signal = tech_data["rsi"].get("signal", "خنثی")
            if rsi_signal == "اشباع فروش":
                base_size += 0.05
        
        if "macd" in tech_data:
            macd_trend = tech_data["macd"].get("trend", "خنثی")
            if macd_trend == "صعودی":
                base_size += 0.05
        
        # تنظیم بر اساس سلامت بازار
        health_status = market_health.get("status", "سالم")
        if "پرریسک" in health_status:
            base_size *= 0.5  # نصف کردن پوزیشن
        elif "بسیار سالم" in health_status:
            base_size *= 1.2  # افزایش 20%
        
        final_size = min(base_size * 100, 30)  # حداکثر 30%
        return f"{final_size:.1f}% از سرمایه"
    
    def _calculate_entry_points(self, tech_analysis: Dict, live_price: float) -> List[str]:
        """محاسبه نقاط ورود"""
        if "error" in tech_analysis or not live_price:
            return ["داده ناکافی برای محاسبه نقاط ورود"]
        
        tech_data = tech_analysis.get("technical_analysis", {})
        price_data = tech_data.get("price_action", {})
        
        support = price_data.get("support_level", live_price * 0.9)
        resistance = price_data.get("resistance_level", live_price * 1.1)
        current_price = live_price
        
        entry_points = []
        
        # نقطه ورود محافظه‌کارانه (2% بالاتر از ساپورت)
        conservative_entry = support * 1.02
        if conservative_entry < current_price:
            entry_points.append(f"ورود محافظه‌کارانه: ${conservative_entry:,.2f}")
        
        # نقطه ورود میانه (1% زیر قیمت فعلی)
        middle_entry = current_price * 0.99
        entry_points.append(f"ورود میانه: ${middle_entry:,.2f}")
        
        # نقطه ورود تهاجمی (3% زیر قیمت فعلی)
        aggressive_entry = current_price * 0.97
        entry_points.append(f"ورود تهاجمی: ${aggressive_entry:,.2f}")
        
        return entry_points
    
    def _generate_exit_strategy(self, tech_analysis: Dict, live_price: float) -> Dict:
        """تولید استراتژی خروج"""
        if "error" in tech_analysis or not live_price:
            return {
                "take_profit": "داده ناکافی",
                "stop_loss": "داده ناکافی",
                "risk_reward_ratio": "نامشخص"
            }
        
        tech_data = tech_analysis.get("technical_analysis", {})
        price_data = tech_data.get("price_action", {})
        
        current_price = live_price
        resistance = price_data.get("resistance_level", current_price * 1.1)
        support = price_data.get("support_level", current_price * 0.9)
        
        # Take Profit (5% سود یا مقاومت، هرکدام کمتر باشد)
        take_profit_1 = current_price * 1.05
        take_profit = min(take_profit_1, resistance)
        
        # Stop Loss (3% ضرر یا ساپورت، هرکدام بیشتر باشد)
        stop_loss_1 = current_price * 0.97
        stop_loss = max(stop_loss_1, support)
        
        profit_potential = ((take_profit - current_price) / current_price) * 100
        loss_potential = ((current_price - stop_loss) / current_price) * 100
        
        risk_reward = (take_profit - current_price) / (current_price - stop_loss) if (current_price - stop_loss) > 0 else 1
        
        return {
            "take_profit": f"${take_profit:,.2f} ({profit_potential:.1f}%)",
            "stop_loss": f"${stop_loss:,.2f} ({loss_potential:.1f}%)",
            "risk_reward_ratio": f"{risk_reward:.2f}:1",
            "assessment": "خوب" if risk_reward > 2 else "متوسط" if risk_reward > 1 else "ضعیف"
        }
    
    def _recommend_timeframe(self, tech_analysis: Dict) -> str:
        """توصیه تایم‌فریم معاملاتی"""
        if "error" in tech_analysis:
            return "نامشخص"
        
        tech_data = tech_analysis.get("technical_analysis", {})
        
        if "risk_metrics" in tech_data and "volatility" in tech_data["risk_metrics"]:
            volatility = tech_data["risk_metrics"]["volatility"].get("value", 0)
            
            if volatility > 8:
                return "کوتاه‌مدت (۱-۷ روز) ⚡"
            elif volatility > 4:
                return "میان‌مدت (۱-۴ هفته) 📅"
            else:
                return "بلندمدت (۱+ ماه) 📈"
        
        return "میان‌مدت (۱-۴ هفته) 📅"
    
    def _calculate_confidence(self, tech_analysis: Dict, sentiment: Dict, market_health: Dict) -> float:
        """محاسبه امتیاز اطمینان"""
        confidence = 0.5
        
        if "error" not in tech_analysis:
            confidence += 0.2
        
        if "error" not in sentiment:
            sentiment_conf = sentiment.get("confidence", 0.5)
            confidence += sentiment_conf * 0.2
        
        if "error" not in market_health:
            health_score = market_health.get("health_score", 0.5)
            confidence += health_score * 0.1
        
        return min(confidence, 1.0)
    
    def get_system_status(self) -> Dict:
        """بررسی وضعیت سیستم و حافظه"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "memory_usage": f"{memory.percent}%",
                "available_memory": f"{memory.available / (1024**3):.1f} GB",
                "disk_usage": f"{disk.percent}%",
                "cpu_usage": f"{psutil.cpu_percent()}%",
                "python_version": sys.version,
                "running_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "process_memory": f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB"
            }
        except Exception as e:
            return {"error": f"خطا در بررسی وضعیت سیستم: {e}"}
    
    def comprehensive_analysis(self, top_coins: int = 5) -> Dict:
        """تحلیل جامع بازار"""
        logger.info(f"🧠 شروع تحلیل جامع بازار برای {top_coins} کوین برتر...")
        
        start_time = datetime.now()
        
        try:
            # بارگذاری داده‌های بازار
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
                "risk_assessment": {},
                "system_status": self.get_system_status(),
                "api_status": self.client.get_api_status()
            }
            
            # تحلیل کوین‌های برتر
            coins_data = self.market_data.get("coins", {}).get("result", [])
            analyzed_coins = 0
            
            logger.info(f"🔍 تحلیل {min(top_coins, len(coins_data))} کوین برتر...")
            
            for coin in coins_data[:top_coins]:
                symbol = coin.get('symbol')
                name = coin.get('name', 'نامشخص')
                
                if symbol and analyzed_coins < top_coins:
                    logger.info(f"  📊 تحلیل {name} ({symbol})...")
                    strategy = self.generate_trading_strategy(symbol)
                    
                    if "error" not in strategy:
                        # اضافه کردن اطلاعات پایه کوین
                        strategy["coin_info"] = {
                            "name": name,
                            "rank": coin.get('rank', 0),
                            "market_cap": coin.get('marketCap', 0),
                            "price_change_24h": coin.get('priceChange1d', 0)
                        }
                        
                        results["top_coins_analysis"].append(strategy)
                        analyzed_coins += 1
            
            # ارزیابی ریسک کلی
            results["risk_assessment"] = {
                "market_risk": results["market_health"].get("status", "نامشخص"),
                "sentiment_risk": results["sentiment_analysis"].get("sentiment", "نامشخص"),
                "technical_risk": self._assess_technical_risk(results["top_coins_analysis"]),
                "overall_risk": self._calculate_overall_risk(results)
            }
            
            # توصیه کلی
            health_status = results["market_health"].get("status", "")
            sentiment = results["sentiment_analysis"].get("sentiment", "")
            technical_risk = results["risk_assessment"].get("technical_risk", "متوسط")
            
            if "سالم" in health_status and "مثبت" in sentiment and technical_risk == "پایین":
                results["overall_recommendation"] = "شرایط عالی برای سرمایه‌گذاری 🎯"
            elif "پرریسک" in health_status or "منفی" in sentiment or technical_risk == "بالا":
                results["overall_recommendation"] = "احتیاط - کاهش حجم معاملات ⚠️"
            else:
                results["overall_recommendation"] = "شرایط متوسط - انتخاب‌های محتاطانه 🔄"
            
            # محاسبه مدت زمان تحلیل
            duration = datetime.now() - start_time
            results["analysis_duration"] = f"{duration.total_seconds():.1f} ثانیه"
            
            logger.info(f"✅ تحلیل جامع تکمیل شد (زمان: {duration.total_seconds():.1f} ثانیه)")
            
            # ذخیره نتایج و اضافه به تاریخچه
            self._save_analysis_results(results)
            self.analysis_history.append(results)
            
            # حفظ فقط 10 تحلیل اخیر در حافظه
            if len(self.analysis_history) > 10:
                self.analysis_history = self.analysis_history[-10:]
            
            return results
            
        except Exception as e:
            error_msg = f"❌ خطا در تحلیل جامع: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _assess_technical_risk(self, coins_analysis: List[Dict]) -> str:
        """ارزیابی ریسک فنی بر اساس تحلیل کوین‌ها"""
        if not coins_analysis:
            return "نامشخص"
        
        high_risk_count = sum(1 for coin in coins_analysis if "بالا" in coin.get("risk_level", ""))
        total_coins = len(coins_analysis)
        
        risk_ratio = high_risk_count / total_coins
        
        if risk_ratio > 0.6:
            return "بالا"
        elif risk_ratio > 0.3:
            return "متوسط"
        else:
            return "پایین"
    
    def _calculate_overall_risk(self, results: Dict) -> str:
        """محاسبه ریسک کلی"""
        risk_factors = 0
        total_factors = 3
        
        market_risk = results["risk_assessment"].get("market_risk", "")
        sentiment_risk = results["risk_assessment"].get("sentiment_risk", "")
        technical_risk = results["risk_assessment"].get("technical_risk", "")
        
        if "پرریسک" in market_risk:
            risk_factors += 1
        if "منفی" in sentiment_risk:
            risk_factors += 1
        if "بالا" in technical_risk:
            risk_factors += 1
        
        risk_score = risk_factors / total_factors
        
        if risk_score > 0.66:
            return "بالا 🔴"
        elif risk_score > 0.33:
            return "متوسط 🟡"
        else:
            return "پایین 🟢"
    
    def _save_analysis_results(self, results: Dict):
        """ذخیره نتایج تحلیل"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/analysis/comprehensive_analysis_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 نتایج در {filename} ذخیره شد")
            
            # همچنین آخرین تحلیل رو در فایل جداگانه ذخیره کن
            latest_file = "data/analysis/latest_analysis.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"⚠️ خطا در ذخیره نتایج: {e}")
    
    def print_analysis_summary(self, analysis: Dict):
        """چاپ خلاصه تحلیل"""
        print("\n" + "="*70)
        print("📊 خلاصه تحلیل هوش مصنوعی بازار کریپتو")
        print("="*70)
        
        if "error" in analysis:
            print(f"❌ خطا: {analysis['error']}")
            return
        
        # سلامت بازار
        health = analysis['market_health']
        print(f"\n🏥 سلامت بازار: {health['status']} (امتیاز: {health['health_score']})")
        print(f"😨 شاخص ترس و طمع: {health['fear_greed_index']} - {health.get('fear_greed_label', '')}")
        print(f"💰 تعداد کوین‌های فعال: {health['active_coins']}")
        
        # احساسات
        sentiment = analysis['sentiment_analysis']
        print(f"\n😊 احساسات بازار: {sentiment['sentiment']} (امتیاز: {sentiment['score']})")
        print(f"📰 اخبار مثبت/منفی: {sentiment['positive_news']}/{sentiment['negative_news']}")
        
        # ریسک
        risk = analysis['risk_assessment']
        print(f"\n⚠️ ارزیابی ریسک:")
        print(f"  • بازار: {risk['market_risk']}")
        print(f"  • احساسات: {risk['sentiment_risk']}")
        print(f"  • فنی: {risk['technical_risk']}")
        print(f"  • کلی: {risk['overall_risk']}")
        
        # توصیه کلی
        print(f"\n💡 توصیه کلی: {analysis['overall_recommendation']}")
        
        # تحلیل کوین‌ها
        print(f"\n🎯 استراتژی‌های کوین‌های برتر:")
        for i, coin_analysis in enumerate(analysis['top_coins_analysis'], 1):
            coin_info = coin_analysis.get('coin_info', {})
            coin_name = coin_info.get('name', coin_analysis['coin'])
            
            print(f"\n  {i}. {coin_name} ({coin_analysis['coin']})")
            print(f"     💰 قیمت: {coin_analysis.get('live_price', 'نامشخص')}")
            print(f"     📈 رتبه: {coin_info.get('rank', 'نامشخص')}")
            print(f"     🎯 توصیه: {coin_analysis['recommendation']}")
            print(f"     ⚠️ ریسک: {coin_analysis['risk_level']}")
            print(f"     📊 اطمینان: {coin_analysis.get('confidence_score', 0.5)*100:.1f}%")
            
            # نمایش اولین نقطه ورود
            entry_points = coin_analysis.get('entry_points', [])
            if entry_points and len(entry_points) > 0:
                print(f"     📍 {entry_points[0]}")
        
        # وضعیت سیستم
        system = analysis['system_status']
        print(f"\n🖥️ وضعیت سیستم:")
        print(f"  • حافظه: {system.get('memory_usage', 'نامشخص')}")
        print(f"  • CPU: {system.get('cpu_usage', 'نامشخص')}")
        print(f"  • مدت تحلیل: {analysis.get('analysis_duration', 'نامشخص')}")
        
        print(f"\n⏰ آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

# ایجاد وب سرور Flask برای مانیتورینگ
app = Flask(__name__)

@app.route('/')
def dashboard():
    """داشبورد مانیتورینگ"""
    try:
        ai_analyst = CryptoAIAnalyst()
        analysis = ai_analyst.comprehensive_analysis(top_coins=3)
        system_status = ai_analyst.get_system_status()
        
        return jsonify({
            "status": "online",
            "analysis": analysis,
            "system": system_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """بررسی سلامت سیستم"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/analysis/<symbol>')
def coin_analysis(symbol):
    """تحلیل یک کوین خاص"""
    try:
        ai_analyst = CryptoAIAnalyst()
        ai_analyst.load_market_data()
        analysis = ai_analyst.generate_trading_strategy(symbol.upper())
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/system')
def system_info():
    """اطلاعات سیستم"""
    try:
        ai_analyst = CryptoAIAnalyst()
        return jsonify(ai_analyst.get_system_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    """تابع اصلی برای اجرای مستقیم"""
    print("🚀 راه‌اندازی هوش مصنوعی تحلیلگر کریپتو...")
    
    try:
        # ایجاد تحلیلگر
        ai_analyst = CryptoAIAnalyst()
        
        # نمایش وضعیت سیستم
        system_status = ai_analyst.get_system_status()
        print(f"🖥️ وضعیت سیستم: {system_status}")
        
        # تحلیل جامع
        print("\n🔍 شروع تحلیل جامع بازار...")
        analysis = ai_analyst.comprehensive_analysis(top_coins=5)
        
        if "error" not in analysis:
            # نمایش نتایج
            ai_analyst.print_analysis_summary(analysis)
        else:
            print(f"❌ خطا: {analysis['error']}")
            
    except Exception as e:
        print(f"❌ خطای غیرمنتظره: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # اگر آرگومان --web داده شد، سرور Flask رو راه‌اندازی کن
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        print("🌐 راه‌اندازی سرور Flask...")
        print("📊 دسترسی به داشبورد: http://localhost:5000")
        print("❤️ سلامت سیستم: http://localhost:5000/health")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        # اجرای معمولی
        main()
