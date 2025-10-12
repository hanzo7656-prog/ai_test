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
    return {
        "ram_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "ram_percent": round(psutil.virtual_memory().percent, 2),
        "cpu_percent": round(psutil.cpu_percent(interval=1), 2),
        "neurons": ai_model.neurons,
        "status": "سالم و فعال"
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

if __name__ == '__main__':
    print("🚀 برنامه شروع شد...")
    app.run(host='0.0.0.0', port=5000, debug=False)
