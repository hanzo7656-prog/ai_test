from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import logging
import json
import asyncio

from ai_brain.config.vortex_brain import vortex_brain
from ai_brain.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# ایجاد روتر چت
chat_router = APIRouter()

# مدیریت سشن‌های چت
chat_sessions = {}
user_sessions = {}

class ChatSession:
    """مدیریت سشن چت کاربر"""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages = []
        self.context = {}
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """اضافه کردن پیام به تاریخچه"""
        message = {
            "role": role,  # user یا assistant
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # حفظ فقط آخرین 50 پیام
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """دریافت تاریخچه مکالمه"""
        return self.messages[-limit:] if self.messages else []
    
    def to_dict(self) -> Dict:
        """تبدیل به دیکشنری"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.messages),
            "messages": self.messages[-10:]  # آخرین 10 پیام
        }

def create_session_id(user_id: str) -> str:
    """ایجاد شناسه سشن"""
    return f"chat_{user_id}_{int(time.time())}"

def get_or_create_session(user_id: str, session_id: Optional[str] = None) -> ChatSession:
    """دریافت یا ایجاد سشن چت"""
    if session_id and session_id in chat_sessions:
        session = chat_sessions[session_id]
        if session.user_id == user_id:
            return session
    
    # ایجاد سشن جدید
    new_session_id = create_session_id(user_id)
    session = ChatSession(new_session_id, user_id)
    chat_sessions[new_session_id] = session
    
    # مدیریت سشن‌های کاربر
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    user_sessions[user_id].append(new_session_id)
    
    # حفظ فقط 5 سشن اخیر برای هر کاربر
    if len(user_sessions[user_id]) > 5:
        oldest_session = user_sessions[user_id].pop(0)
        if oldest_session in chat_sessions:
            del chat_sessions[oldest_session]
    
    return session

@chat_router.post("/send")
async def send_chat_message(
    message: str,
    user_id: str = "anonymous",
    session_id: Optional[str] = None
):
    """ارسال پیام در چت"""
    try:
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="پیام نمی‌تواند خالی باشد")
        
        # دریافت یا ایجاد سشن
        session = get_or_create_session(user_id, session_id)
        
        # اضافه کردن پیام کاربر به تاریخچه
        session.add_message("user", message)
        
        # پردازش توسط هوش مصنوعی
        start_time = time.time()
        ai_response = await vortex_brain.process_query(message, user_id)
        response_time = time.time() - start_time
        
        # استخراج پاسخ
        response_text = ai_response.get('response', 'پاسخی دریافت نشد')
        success = ai_response.get('success', False)
        
        # اضافه کردن پاسخ هوش مصنوعی به تاریخچه
        session.add_message("assistant", response_text, {
            "response_time": round(response_time, 3),
            "intent": ai_response.get('intent'),
            "confidence": ai_response.get('confidence'),
            "success": success
        })
        
        return {
            "success": True,
            "session_id": session.session_id,
            "response": response_text,
            "response_time": round(response_time, 3),
            "message_id": len(session.messages),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "intent": ai_response.get('intent'),
                "confidence": ai_response.get('confidence'),
                "context_used": len(session.get_conversation_history()) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در ارسال پیام: {e}")
        raise HTTPException(status_code=500, detail=f"خطا در پردازش پیام: {str(e)}")

@chat_router.get("/sessions")
async def get_user_sessions(user_id: str, limit: int = 5):
    """دریافت سشن‌های کاربر"""
    try:
        sessions = []
        user_session_ids = user_sessions.get(user_id, [])
        
        for session_id in user_session_ids[-limit:]:  # آخرین سشن‌ها
            if session_id in chat_sessions:
                session = chat_sessions[session_id]
                sessions.append(session.to_dict())
        
        return {
            "user_id": user_id,
            "total_sessions": len(user_session_ids),
            "sessions": sessions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در دریافت سشن‌ها: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.get("/history")
async def get_chat_history(session_id: str, limit: int = 20):
    """دریافت تاریخچه مکالمه"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="سشن یافت نشد")
        
        session = chat_sessions[session_id]
        messages = session.messages[-limit:] if session.messages else []
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "total_messages": len(session.messages),
            "messages": messages,
            "session_created": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطا در دریافت تاریخچه: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """حذف سشن چت"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="سشن یافت نشد")
        
        session = chat_sessions[session_id]
        user_id = session.user_id
        
        # حذف از مدیریت سشن‌ها
        del chat_sessions[session_id]
        if user_id in user_sessions and session_id in user_sessions[user_id]:
            user_sessions[user_id].remove(session_id)
        
        return {
            "success": True,
            "message": "سشن با موفقیت حذف شد",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطا در حذف سشن: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.get("/suggestions")
async def get_chat_suggestions(user_id: str = "anonymous"):
    """دریافت پیشنهادات سوال"""
    suggestions = [
        "قیمت بیتکوین چنده؟",
        "اخبار جدید ارزهای دیجیتال رو بگو",
        "وضعیت سیستم چطوره؟",
        "شاخص ترس و طمع بازار چنده؟",
        "لیست 10 ارز برتر رو نشون بده",
        "تحلیل تکنیکال اتریوم رو بگو",
        "وضعیت کش سیستم چطوره؟",
        "داده‌های خام بیتکوین رو بفرست"
    ]
    
    return {
        "user_id": user_id,
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat()
    }

# WebSocket برای چت real-time
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

@chat_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket برای چت real-time"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # پردازش پیام
            response = await send_chat_message(
                message=message_data.get("message", ""),
                user_id=user_id,
                session_id=message_data.get("session_id")
            )
            
            # ارسال پاسخ
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"❌ خطا در WebSocket: {e}")
        manager.disconnect(user_id)

# پاک‌سازی سشن‌های قدیمی
async def cleanup_old_sessions():
    """پاک‌سازی سشن‌های قدیمی"""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in chat_sessions.items():
                if current_time - session.last_activity > timedelta(hours=24):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = chat_sessions[session_id]
                user_id = session.user_id
                
                del chat_sessions[session_id]
                if user_id in user_sessions and session_id in user_sessions[user_id]:
                    user_sessions[user_id].remove(session_id)
            
            if expired_sessions:
                logger.info(f"🧹 پاک‌سازی {len(expired_sessions)} سشن منقضی شده")
            
            await asyncio.sleep(3600)  # هر 1 ساعت
            
        except Exception as e:
            logger.error(f"❌ خطا در پاک‌سازی سشن‌ها: {e}")
            await asyncio.sleep(300)

@chat_router.on_event("startup")
async def startup_event():
    """رویداد راه‌اندازی"""
    asyncio.create_task(cleanup_old_sessions())

@chat_router.on_event("shutdown")
async def shutdown_event():
    """رویداد خاموش‌سازی"""
    # پاک‌سازی منابع اگر نیاز باشد
    pass
