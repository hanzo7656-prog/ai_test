from flask import Flask, jsonify
import psutil
import platform
import multiprocessing

app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 AI Test Server is Working!"

@app.route('/specs')
def specs():
    return jsonify({
        "cpu_cores": multiprocessing.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 1),
        "os": f"{platform.system()} {platform.release()}"
    })

@app.route('/ai-test')
def ai_test():
    # یک مدل AI ساده برای تست
    def simple_predict(price):
        if price < 100:
            return "BUY", 0.8
        else:
            return "HOLD", 0.6
    
    signal, confidence = simple_predict(85)
    return jsonify({
        "signal": signal,
        "confidence": confidence,
        "message": "Simple AI is working!"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
