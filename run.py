# run.py - فایل اجرایی
from main import app
import uvicorn
import logging
import os

# تنظیمات logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # گرفتن پورت از متغیر محیطی رندر
    PORT = int(os.environ.get("PORT", 10000))
    
    logger.info(f"🚀 Starting Crypto AI Trading Server on port {PORT}...")
    logger.info(f"📚 API Documentation: http://0.0.0.0:{PORT}/api/docs")
    logger.info(f"❤️ Health Check: http://0.0.0.0:{PORT}/api/health")
    logger.info(f"🔍 System Info: http://0.0.0.0:{PORT}/api/info")
    logger.info(f"🌐 Live URL: https://ai-test-grzf.onrender.com")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True,
        workers=1
    )
