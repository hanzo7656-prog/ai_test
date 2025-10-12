from flask import Flask, jsonify
import psutil
import os

app = Flask(__name__)

# یک مدل فوق‌العاده ساده
class SimpleAI:
    def __init__(self):
        self.neurons = 100
        print(f"🤖 مدل AI با {self.neurons} نورون راه‌اندازی شد")
    
    def predict(self):
        # یک محاسبه سکه برای تست
        return "AI is thinking..."

ai_model = SimpleAI()

@app.route('/')
def home():
    return '''
    <h1>🧠 AI Crypto Analyzer - LIVE</h1>
    <p>مدل با ۱۰۰ نورون فعال است</p>
    <a href="/health">بررسی سلامت</a> | 
    <a href="/predict">تست AI</a>
    '''

@app.route('/health')
def health():
    process = psutil.Process(os.getpid())
    
    return jsonify({
        "status": "✅ سالم و فعال",
        "ram_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "ram_percent": round(psutil.virtual_memory().percent, 2),
        "cpu_percent": round(psutil.cpu_percent(interval=1), 2),
        "neurons": ai_model.neurons,
        "message": "همه چیز اوکی هست! 🚀"
    })

@app.route('/predict')
def predict():
    result = ai_model.predict()
    
    return jsonify({
        "prediction": result,
        "neurons_used": ai_model.neurons,
        "status": "پیش‌بینی انجام شد"
    })

if __name__ == '__main__':
    print("🚀 برنامه شروع شد...")
    app.run(host='0.0.0.0', port=5000, debug=False)
