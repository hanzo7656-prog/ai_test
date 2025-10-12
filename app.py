from flask import Flask, jsonify, render_template
import psutil
import os
import time

app = Flask(__name__)

class SimpleAI:
    def __init__(self):
        self.neurons = 100
        print(f"🤖 مدل AI با {self.neurons} نورون راه‌اندازی شد")
    
    def predict(self):
        return "AI is thinking..."

ai_model = SimpleAI()

def get_system_info():
    process = psutil.Process(os.getpid())
    
    # روش‌های مختلف اندازه‌گیری RAM
    try:
        # روش ۱: استفاده از process memory
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # روش ۲: استفاده از virtual memory
        virtual_memory = psutil.virtual_memory()
        
        # روش ۳: استفاده از memory_usage (اگر available باشه)
        memory_percent = process.memory_percent()
        
        return {
            "ram_used_mb": round(process_memory, 2),
            "ram_percent": round(memory_percent, 2),
            "total_ram_mb": round(virtual_memory.total / 1024 / 1024, 2),
            "available_ram_mb": round(virtual_memory.available / 1024 / 1024, 2),
            "virtual_memory_percent": round(virtual_memory.percent, 2),
            "cpu_percent": round(psutil.cpu_percent(interval=1), 2),
            "neurons": ai_model.neurons,
            "status": "سالم و فعال"
        }
    except Exception as e:
        return {
            "ram_used_mb": 0,
            "ram_percent": 0,
            "total_ram_mb": 0,
            "available_ram_mb": 0,
            "virtual_memory_percent": 0,
            "cpu_percent": 0,
            "neurons": ai_model.neurons,
            "status": f"خطا در اندازه‌گیری: {str(e)}"
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify(get_system_info())

@app.route('/predict')
def predict():
    start_time = time.time()
    result = ai_model.predict()
    processing_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "prediction": result,
        "processing_time_ms": processing_time,
        "neurons_used": ai_model.neurons,
        "message": "پیش‌بینی انجام شد"
    })

# اضافه کردن endpoint جدید برای اطلاعات دقیق
@app.route('/debug/ram')
def debug_ram():
    process = psutil.Process(os.getpid())
    virtual_memory = psutil.virtual_memory()
    
    return jsonify({
        "process_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "process_vms_mb": round(process.memory_info().vms / 1024 / 1024, 2),
        "process_memory_percent": round(process.memory_percent(), 2),
        "virtual_memory_total_mb": round(virtual_memory.total / 1024 / 1024, 2),
        "virtual_memory_available_mb": round(virtual_memory.available / 1024 / 1024, 2),
        "virtual_memory_used_mb": round(virtual_memory.used / 1024 / 1024, 2),
        "virtual_memory_percent": round(virtual_memory.percent, 2),
        "system_wide_info": str(psutil.virtual_memory())
    })

if __name__ == '__main__':
    print("🚀 برنامه شروع شد...")
    # چاپ اطلاعات RAM هنگام شروع
    info = get_system_info()
    print(f"📊 اطلاعات RAM: {info}")
    app.run(host='0.0.0.0', port=5000, debug=False)
