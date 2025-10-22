# 📁 run.py

import uvicorn
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # بارگذاری متغیرهای محیطی
    load_dotenv()
    
    # اجرای سرور - مستقیماً از مسیر درست
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info")
    )
