# run.py
from main import app
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
