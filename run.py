# run.py - فایل اجرایی
from main import app
import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))
    
    logger.info(f"🚀 Starting CryptoAI Server on port {PORT}")
    logger.info(f"🌐 Health Check: http://0.0.0.0:{PORT}/api/health")
    logger.info(f"📊 System Status: http://0.0.0.0:{PORT}/api/system/status")
    logger.info(f"🤖 AI Scan: http://0.0.0.0:{PORT}/api/ai/scan")
    logger.info(f"📈 Technical Analysis: http://0.0.0.0:{PORT}/api/ai/technical/analysis")
    logger.info(f"🔧 System Debug: http://0.0.0.0:{PORT}/api/system/debug")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True,
        reload=False
    )
