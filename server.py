# serve.py
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # فعال کردن CORS

# مسیر فایل‌های استاتیک
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def serve_index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

# endpoint سلامت برای تست
@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "Vortex AI Frontend",
        "timestamp": "2025-10-21T18:15:00"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
