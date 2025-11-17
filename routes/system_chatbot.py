# routes/system_chatbot.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
import re
import json
from typing import Dict, List, Any
from debug_system.storage.redis_manager import redis_manager  # Redis Manager تو

chatbot_router = APIRouter(prefix="/api/chatbot", tags=["System Chatbot"])

class VortexAIChatbot:
    def __init__(self):
        self.db_name = "mother_a"
        
        # 🎯 دستورات و اندپوینت‌های سیستم
        self.commands = {
            # دستورات سلامت سیستم
            "سلامت": {
                "endpoint": "/api/health/status",
                "params": {"detail": "basic"},
                "description": "بررسی سلامت کلی سیستم"
            },
            "کش": {
                "endpoint": "/api/health/cache", 
                "params": {"view": "status"},
                "description": "وضعیت سیستم کش"
            },
            "هشدار": {
                "endpoint": "/api/health/debug",
                "params": {"view": "alerts"}, 
                "description": "لیست هشدارهای فعال"
            },
            "منابع": {
                "endpoint": "/api/health/metrics",
                "params": {"type": "system"},
                "description": "مصرف منابع سیستم"
            },
            "کارگر": {
                "endpoint": "/api/health/workers", 
                "params": {"metric": "status"},
                "description": "وضعیت background workers"
            },
            
            # دستورات کوین‌ها
            "بیتکوین": {
                "endpoint": "/api/coins/details/bitcoin",
                "params": {},
                "description": "اطلاعات بیت‌کوین"
            },
            "اتریوم": {
                "endpoint": "/api/coins/details/ethereum", 
                "params": {},
                "description": "اطلاعات اتریوم"
            },
            "لیست ارز": {
                "endpoint": "/api/coins/list",
                "params": {"limit": "10"},
                "description": "لیست 10 ارز برتر"
            },
            
            # دستورات اخبار
            "اخبار": {
                "endpoint": "/api/news/all",
                "params": {"limit": "5"},
                "description": "آخرین اخبار کریپتو"
            },
            "ترس و طمع": {
                "endpoint": "/api/insights/fear-greed", 
                "params": {},
                "description": "شاخص ترس و طمع"
            },
            
            # دستورات صرافی
            "صرافی": {
                "endpoint": "/api/exchanges/list",
                "params": {},
                "description": "لیست صرافی‌ها"
            }
        }
        
        # 🧠 الگوهای یادگیری
        self.learning_patterns = {}
    
    def understand_command(self, user_message: str) -> Dict[str, Any]:
        """درک دستور کاربر و پیدا کردن اندپوینت مناسب"""
        user_message = user_message.lower().strip()
        
        # اول چک کن آیا الگوی یادگرفته‌شده exist داره
        learned_response = self._check_learned_patterns(user_message)
        if learned_response:
            return learned_response
        
        # جستجو در دستورات اصلی
        for keyword, config in self.commands.items():
            if keyword in user_message:
                return {
                    "command": keyword,
                    "endpoint": config["endpoint"],
                    "params": config["params"],
                    "confidence": 0.9,
                    "type": "direct_match"
                }
        
        # اگر مستقیم پیدا نشد، از الگوهای هوشمند استفاده کن
        smart_match = self._smart_pattern_match(user_message)
        if smart_match:
            return smart_match
        
        # اگر چیزی پیدا نشد
        return {
            "command": "unknown",
            "confidence": 0.0,
            "suggestions": self._get_suggestions(user_message)
        }
    
    def _smart_pattern_match(self, message: str) -> Dict[str, Any]:
        """الگوی هوشمند برای درک بهتر سوالات"""
        
        # الگوهای سلامت سیستم
        if any(word in message for word in ["وضعیت", "سلامتی", "سیستم", "چطوره"]):
            if any(word in message for word in ["کش", "کَش"]):
                return self.commands["کش"]
            elif any(word in message for word in ["هشدار", "خطا"]):
                return self.commands["هشدار"] 
            elif any(word in message for word in ["منابع", "رم", "سیپییو"]):
                return self.commands["منابع"]
            else:
                return self.commands["سلامت"]
        
        # الگوهای قیمت ارز
        if any(word in message for word in ["قیمت", "نرخ", "ارزش"]):
            if "بیت" in message or "btc" in message:
                return self.commands["بیتکوین"]
            elif "اتری" in message or "eth" in message:
                return self.commands["اتریوم"]
            elif "لیست" in message or "ارز" in message:
                return self.commands["لیست ارز"]
        
        # الگوهای اخبار
        if any(word in message for word in ["خبر", "اخبار", "تازه"]):
            return self.commands["اخبار"]
        
        return None
    
    def _check_learned_patterns(self, message: str) -> Dict[str, Any]:
        """بررسی الگوهای یادگرفته‌شده از قبل"""
        # اینجا می‌تونی از Redis برای ذخیره الگوهای یادگرفته‌شده استفاده کنی
        pattern_key = f"chatbot:learned_patterns:{hash(message)}"
        learned, _ = redis_manager.get(self.db_name, pattern_key)
        
        if learned:
            return learned
        return None
    
    def _get_suggestions(self, message: str) -> List[str]:
        """پیشنهاد دستورات مشابه"""
        words = set(message.split())
        suggestions = []
        
        for cmd in self.commands.keys():
            cmd_words = set(cmd.split())
            if words.intersection(cmd_words):
                suggestions.append(cmd)
        
        return suggestions[:3]
    
    async def learn_from_interaction(self, user_message: str, api_response: Dict, success: bool = True):
        """یادگیری از تعامل کاربر"""
        if success and api_response.get("command") != "unknown":
            # ذخیره الگوی موفق
            pattern_data = {
                "user_message": user_message,
                "command": api_response["command"],
                "endpoint": api_response["endpoint"],
                "timestamp": datetime.now().isoformat(),
                "success_count": 1
            }
            
            pattern_key = f"chatbot:learned_patterns:{hash(user_message)}"
            redis_manager.set(self.db_name, pattern_key, pattern_data, 30*24*3600)
    
    def format_response(self, command: str, api_data: Dict) -> str:
        """فرمت‌دهی پاسخ به صورت خوانا برای کاربر"""
        
        if command == "سلامت":
            health_score = api_data.get("health_score", 0)
            status = "🟢 عالی" if health_score > 80 else "🟡 قابل قبول" if health_score > 60 else "🔴 نیاز توجه"
            return f"🏥 وضعیت سیستم: {status}\n• امتیاز سلامت: {health_score}%\n• وضعیت: {api_data.get('status', 'نامشخص')}"
        
        elif command == "کش":
            cache_health = api_data.get("health", {})
            dbs_connected = cache_health.get("cloud_resources", {}).get("databases_connected", 0)
            return f"💾 سیستم کش: {dbs_connected}/5 دیتابیس متصل\n• امتیاز: {cache_health.get('health_score', 0)}%"
        
        elif command == "هشدار":
            alerts = api_data.get("active_alerts", [])
            if not alerts:
                return "✅ هیچ هشدار فعالی وجود ندارد"
            else:
                critical = len([a for a in alerts if a.get('level') == 'CRITICAL'])
                return f"🚨 {len(alerts)} هشدار فعال\n• 🔴 {critical} هشدار بحرانی"
        
        elif command == "منابع":
            system = api_data.get("system", {})
            cpu = system.get("cpu", {}).get("usage_percent", 0)
            memory = system.get("memory", {}).get("usage_percent", 0)
            return f"⚡ مصرف منابع:\n• پردازنده: {cpu}%\n• حافظه: {memory}%"
        
        elif command == "بیتکوین":
            price = api_data.get("data", {}).get("price", 0)
            change = api_data.get("data", {}).get("price_change_24h", 0)
            trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            return f"₿ بیت‌کوین: ${price:,.2f}\n• تغییر 24h: {trend} {abs(change)}%"
        
        elif command == "لیست ارز":
            coins = api_data.get("data", [])
            if not coins:
                return "❌ اطلاعات ارزی دریافت نشد"
            
            top_coins = coins[:3]  # 3 ارز اول
            response = "🏆 برترین ارزها:\n"
            for coin in top_coins:
                response += f"• {coin.get('symbol', '')}: ${coin.get('price', 0):,.2f}\n"
            return response.strip()
        
        elif command == "اخبار":
            news = api_data.get("data", [])
            if not news:
                return "📰 خبری یافت نشد"
            return f"📰 {len(news)} خبر جدید دریافت شد\n• اولین خبر: {news[0].get('title', '')}"
        
        else:
            return f"📊 اطلاعات دریافت شد: {command}"

# نمونه اصلی
vortex_bot = VortexAIChatbot()

@chatbot_router.post("/ask")
async def ask_bot(question: str, user_id: str = "default"):
    """سوال از چت بات سیستم"""
    
    # 1. درک دستور کاربر
    command_info = vortex_bot.understand_command(question)
    
    if command_info["command"] == "unknown":
        return {
            "success": False,
            "answer": "❌ متوجه سوال شما نشدم. می‌تونید در مورد این موارد بپرسید: سلامت سیستم، وضعیت کش، قیمت ارزها، اخبار",
            "suggestions": command_info.get("suggestions", [])
        }
    
    try:
        # 2. فراخوانی API مربوطه
        # اینجا باید کد واقعی فراخوانی API رو بنویسی
        # response = await call_api(command_info["endpoint"], command_info["params"])
        
        # برای نمونه، یک پاسخ ساختگی:
        sample_responses = {
            "سلامت": {"health_score": 95, "status": "healthy"},
            "کش": {"health": {"cloud_resources": {"databases_connected": 5}, "health_score": 90}},
            "بیتکوین": {"data": {"price": 45000, "price_change_24h": 2.5}},
            "لیست ارز": {"data": [
                {"symbol": "BTC", "price": 45000},
                {"symbol": "ETH", "price": 2500},
                {"symbol": "SOL", "price": 100}
            ]}
        }
        
        api_data = sample_responses.get(command_info["command"], {})
        
        # 3. فرمت‌دهی پاسخ
        formatted_answer = vortex_bot.format_response(command_info["command"], api_data)
        
        # 4. یادگیری از این تعامل
        await vortex_bot.learn_from_interaction(question, command_info, success=True)
        
        return {
            "success": True,
            "answer": formatted_answer,
            "command": command_info["command"],
            "endpoint": command_info.get("endpoint"),
            "confidence": command_info.get("confidence", 0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "answer": f"❌ خطا در دریافت اطلاعات: {str(e)}",
            "error": str(e)
        }

@chatbot_router.get("/commands")
async def get_available_commands():
    """دریافت لیست دستورات موجود"""
    commands_list = []
    for cmd, config in vortex_bot.commands.items():
        commands_list.append({
            "command": cmd,
            "description": config["description"],
            "endpoint": config["endpoint"]
        })
    
    return {"commands": commands_list}
