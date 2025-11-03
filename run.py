# run.py - فایل اجرایی
from main import app
import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))
    
    logger.info(f"🚀 Starting Server on port {PORT}")
    logger.info(f"🌐 Health Check: http://0.0.0.0:{PORT}/api/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )
