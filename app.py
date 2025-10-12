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
        # یک محاسبه ساده برای تست CPU
        result = 0
        for i in range(1000):
            result += i * 0.1
        return f"AI calculated: {result}"

ai_model = SimpleAI()

def get_real_system_info():
    process = psutil.Process(os.getpid())
    
    try:
        # اندازه‌گیری CPU با روش مطمئن‌تر
        # روش ۱: process-specific با interval
        process_cpu = process.cpu_percent(interval=0.5)
        
        # روش ۲: system-wide برای مقایسه
        system_cpu = psutil.cpu_percent(interval=0.5)
        
        # روش ۳: انجام یک محاسبه کوچک برای تست
        test_start = time.time()
        test_calculation = sum(i * i for i in range(10000))
        test_duration = time.time() - test_start
        
        # اگر process_cpu هنوز 0 بود، از system_cpu استفاده کن
        final_cpu = process_cpu if process_cpu > 0 else system_cpu
        
        # اگر باز هم 0 بود، از test duration تخمین بزن
        if final_cpu == 0 and test_duration > 0:
            # تخمین بر اساس زمان پردازش
            estimated_cpu = min((test_duration / 0.5) * 100, 100)
            final_cpu = round(estimated_cpu, 2)
        
        # RAM measurement (همان قبلی)
        process_memory_mb = process.memory_info().rss / 1024 / 1024
        estimated_ram_percent = (process_memory_mb / 512) * 100  # فرض 512MB برای رایگان
        
        return {
            # RAM اطلاعات
            "ram_used_mb": round(process_memory_mb, 2),
            "ram_percent": round(estimated_ram_percent, 2),
            "total_ram_mb": 512,
            
            # CPU اطلاعات با روش‌های مختلف
            "cpu_percent": round(final_cpu, 2),
            "cpu_percent_process": round(process_cpu, 2),
            "cpu_percent_system": round(system_cpu, 2),
            "test_duration_ms": round(test_duration * 1000, 2),
            
            # اطلاعات مدل
            "neurons": ai_model.neurons,
            "status": "سالم و فعال",
            "measurement_note": "CPU measurement optimized"
        }
        
    except Exception as e:
        return {
            "ram_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "ram_percent": 5.0,
            "total_ram_mb": 512,
            "cpu_percent": 0.5,  # مقدار پیش‌فرض
            "neurons": ai_model.neurons,
            "status": "سالم و فعال (فال‌بک)",
            "error": str(e)
        }

# اضافه کردن endpoint برای تست CPU
@app.route('/test-cpu')
def test_cpu():
    """انجام یک محاسبه سنگین‌تر برای تست CPU"""
    start_time = time.time()
    
    # محاسبه سنگین‌تر
    result = 0
    for i in range(1000000):
        result += i * 0.0001
    
    duration = (time.time() - start_time) * 1000
    
    return jsonify({
        "test_result": round(result, 4),
        "processing_time_ms": round(duration, 2),
        "cpu_usage_note": "محاسبه تستی انجام شد"
    })



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify(get_real_system_info())

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

# endpoint جدید برای اطلاعات دقیق
@app.route('/debug/system')
def debug_system():
    process = psutil.Process(os.getpid())
    info = get_real_system_info()
    
    debug_info = {
        **info,
        "process_memory_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "process_memory_vms_mb": round(process.memory_info().vms / 1024 / 1024, 2),
        "process_cpu_times": str(process.cpu_times()),
        "virtual_memory_total": round(psutil.virtual_memory().total / 1024 / 1024, 2),
        "virtual_memory_used": round(psutil.virtual_memory().used / 1024 / 1024, 2),
        "virtual_memory_percent": psutil.virtual_memory().percent,
        "system_cpu_percent": psutil.cpu_percent(interval=1)
    }
    
    return jsonify(debug_info)

if __name__ == '__main__':
    print("🚀 برنامه شروع شد...")
    # چاپ اطلاعات واقعی هنگام شروع
    info = get_real_system_info()
    print(f"📊 اطلاعات واقعی سیستم: {info}")
    app.run(host='0.0.0.0', port=5000, debug=False)
