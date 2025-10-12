from flask import Flask, jsonify
import psutil
import os
import time
import numpy as np

app = Flask(__name__)

class SimpleNeuralNetwork:
    def __init__(self, input_size=10, hidden_size=100, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size  # 100 نورون
        self.output_size = output_size
        
        # وزن‌های تصادفی ساده
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32)
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float32)
        
        print(f"✅ شبکه عصبی ساخته شد: {hidden_size} نورون")
    
    def predict(self, X):
        # پیش‌بینی ساده
        hidden = np.dot(X, self.W1)
        hidden_relu = np.maximum(0, hidden)  # ReLU
        output = np.dot(hidden_relu, self.W2)
        return output

# ایجاد مدل
model = SimpleNeuralNetwork()

def get_system_info():
    """اطلاعات سیستم رو برمی‌گرداند"""
    process = psutil.Process(os.getpid())
    
    return {
        "ram_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "ram_percent": round(psutil.virtual_memory().percent, 2),
        "cpu_percent": round(psutil.cpu_percent(interval=1), 2),
        "model_neurons": model.hidden_size,
        "model_weights": f"{model.W1.size + model.W2.size} پارامتر",
        "status": "زنده و فعال 🚀"
    }

@app.route('/')
def home():
    return """
    <h1>🤖 AI Crypto Analyzer - TEST MODE</h1>
    <p>شبکه عصبی با ۱۰۰ نورون فعال است</p>
    <a href="/health">سلامت سیستم</a> | 
    <a href="/predict">تست پیش‌بینی</a> |
    <a href="/stress">تست فشار</a>
    """

@app.route('/health')
def health():
    """بررسی سلامت سیستم"""
    info = get_system_info()
    return jsonify(info)

@app.route('/predict')
def predict():
    """تست پیش‌بینی ساده"""
    start_time = time.time()
    
    # داده تستی
    X_test = np.random.randn(1, model.input_size).astype(np.float32)
    
    # پیش‌بینی
    prediction = model.predict(X_test)
    
    # محاسبه زمان
    processing_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "prediction": float(prediction[0][0]),
        "processing_time_ms": processing_time,
        "neurons_used": model.hidden_size,
        "message": "پیش‌بینی انجام شد",
        **get_system_info()
    })

@app.route('/stress')
def stress_test():
    """تست فشار روی CPU و RAM"""
    start_time = time.time()
    
    # تست سنگین‌تر
    results = []
    for i in range(100):  # 100 بار پیش‌بینی
        X_test = np.random.randn(1, model.input_size).astype(np.float32)
        pred = model.predict(X_test)
        results.append(float(pred[0][0]))
    
    total_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "stress_test_iterations": 100,
        "total_time_ms": total_time,
        "avg_time_per_prediction_ms": round(total_time / 100, 2),
        "predictions_range": f"{min(results):.4f} to {max(results):.4f}",
        **get_system_info()
    })

if __name__ == '__main__':
    print("🚀 شروع برنامه...")
    print("📊 اطلاعات اولیه سیستم:")
    info = get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
