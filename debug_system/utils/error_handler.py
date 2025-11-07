import logging
import traceback
from typing import Dict, Any, Optional, Callable
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime

logger = logging.getLogger(__name__)

class APIError(Exception):
    """خطای سفارشی برای API"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Dict[str, Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ErrorHandler:
    def __init__(self):
        self.error_handlers = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """ثبت هندلرهای پیش‌فرض"""
        self.register_handler(APIError, self._handle_api_error)
        self.register_handler(HTTPException, self._handle_http_exception)
        self.register_handler(Exception, self._handle_generic_error)
    
    def register_handler(self, exception_type: type, handler: Callable):
        """ثبت هندلر برای نوع خطای خاص"""
        self.error_handlers[exception_type] = handler
        logger.debug(f"✅ Registered error handler for {exception_type.__name__}")
    
    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """مدیریت خطاهای全局"""
        try:
            # پیدا کردن هندلر مناسب
            handler = None
            for exc_type in type(exc).__mro__:
                if exc_type in self.error_handlers:
                    handler = self.error_handlers[exc_type]
                    break
            
            if handler:
                return await handler(request, exc)
            else:
                return await self._handle_generic_error(request, exc)
                
        except Exception as e:
            # fallback برای خطا در خود هندلر خطا
            logger.critical(f"💥 Error in error handler: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def _handle_api_error(self, request: Request, exc: APIError) -> JSONResponse:
        """هندلر خطاهای API سفارشی"""
        logger.warning(f"🚨 API Error: {exc.error_code} - {exc.message}")
        
        error_response = {
            "error": exc.error_code,
            "message": exc.message,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
        
        if exc.details:
            error_response["details"] = exc.details
        
        # لاگ کردن جزئیات بیشتر برای خطاهای سرور
        if exc.status_code >= 500:
            logger.error(f"🔴 Server Error: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """هندلر خطاهای HTTP"""
        logger.info(f"🌐 HTTP Error: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
        )
    
    async def _handle_generic_error(self, request: Request, exc: Exception) -> JSONResponse:
        """هندلر خطاهای عمومی"""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.error(
            f"💥 Unhandled Error [{error_id}]: {str(exc)}\n"
            f"Path: {request.url.path}\n"
            f"Method: {request.method}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        # در production جزئیات خطا را نشان نده
        is_development = True  # این باید از environment variable خوانده شود
        
        error_response = {
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "error_id": error_id,
            "path": request.url.path
        }
        
        if is_development:
            error_response["debug_info"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc().split('\n')
            }
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    def create_error_response(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """ایجاد response خطای ساختاریافته"""
        return {
            "error": error_code,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
    
    def wrap_async_function(self, func: Callable) -> Callable:
        """wrap کردن تابع async برای مدیریت خطا"""
        async def wrapped(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                # ایجاد یک request ساختگی برای هندلر
                class MockRequest:
                    def __init__(self):
                        self.url.path = getattr(args[0], 'path', '/unknown') if args else '/unknown'
                        self.method = getattr(args[0], 'method', 'GET') if args else 'GET'
                
                mock_request = MockRequest()
                return await self.handle_exception(mock_request, exc)
        
        return wrapped

# ایجاد نمونه گلوبال
error_handler = ErrorHandler()
