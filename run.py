from main import app
import uvicorn
import logging

# تنظیمات logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting Crypto AI Trading Server on port 8000...")
    logger.info("📚 API Documentation: http://0.0.0.0:8000/docs")
    logger.info("❤️ Health Check: http://0.0.0.0:8000/health")
    logger.info("🔍 System Info: http://0.0.0.0:8000/api/info")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        access_log=True
    )
