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
        # اندازه‌گیری CPU واقعی
        process_cpu = process.cpu_percent(interval=1.0)
        
        # برای RAM: استفاده از روش ترکیبی
        process_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # تخمین RAM واقعی بر اساس اطلاعات Render Free Tier
        estimated_total_ram_mb = 512  # پلن رایگان Render
        estimated_ram_percent = (process_memory_mb / estimated_total_ram_mb) * 100
        
        # اگر psutil درصد منطقی‌تری میده از اون استفاده کن
        virtual_memory = psutil.virtual_memory()
        psutil_ram_percent = virtual_memory.percent
        
        # انتخاب روش بهتر برای RAM
        if 0 < psutil_ram_percent < 100:  # اگر psutil عدد منطقی میده
            final_ram_percent = psutil_ram_percent
            final_total_ram_mb = virtual_memory.total / 1024 / 1024
        else:  # در غیر این صورت از تخمین استفاده کن
            final_ram_percent = estimated_ram_percent
            final_total_ram_mb = estimated_total_ram_mb
        
        # نرمال‌سازی CPU برای نمایش بهتر
        normalized_cpu = min(process_cpu, 100)  # حداکثر 100%
        
        return {
            # RAM اطلاعات
            "ram_used_mb": round(process_memory_mb, 2),
            "ram_percent": round(final_ram_percent, 2),
            "total_ram_mb": round(final_total_ram_mb, 2),
            
            # CPU اطلاعات
            "cpu_percent": round(normalized_cpu, 2),
            "cpu_count": psutil.cpu_count(),
            
            # اطلاعات مدل
            "neurons": ai_model.neurons,
            "status": "سالم و فعال",
            
            # متادیتا برای دیباگ
            "measurement_method": "optimized_container_measurement"
        }
        
    except Exception as e:
        # فال‌بک به روش ساده اگر خطا داشتیم
        return {
            "ram_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "ram_percent": 5.0,  # مقدار پیش‌فرض منطقی
            "total_ram_mb": 512,
            "cpu_percent": round(process.cpu_percent(interval=1), 2),
            "cpu_count": 1,
            "neurons": ai_model.neurons,
            "status": "سالم و فعال (فال‌بک)",
            "measurement_method": "fallback_simple",
            "error": str(e)
        }

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
