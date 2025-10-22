# 📁 src/api/middleware.py

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter(BaseHTTPMiddleware):
    """میدلور محدودیت نرخ درخواست"""
    
    def __init__(self, app, requests_per_minute: int = 100, burst_limit: int = 20):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # پاک کردن درخواست‌های قدیمی
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60  # فقط 60 ثانیه اخیر
            ]
        
        # بررسی محدودیت نرخ
        if client_ip in self.requests:
            if len(self.requests[client_ip]) >= self.burst_limit:
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limit exceeded. Try again later."
                )
            
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=429, 
                    detail="Too many requests. Please wait a minute."
                )
        
        # ثبت درخواست جدید
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response

class SecurityHeaders(BaseHTTPMiddleware):
    """میدلور هدرهای امنیتی"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # اضافه کردن هدرهای امنیتی
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

async def verify_api_key(request: Request):
    """احراز هویت API Key"""
    api_key = request.headers.get("X-API-Key")
    expected_key = "your-secret-api-key"  # در production از environment variables استفاده شود
    
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )
    
    return api_key
