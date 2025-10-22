# 📁 main.py

import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # در production همیشه false
        log_level="info"
    )
