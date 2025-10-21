# app.py
import os
import logging
from flask import Flask, jsonify
from datetime import datetime

# تنظیمات لاگ برای Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

app = Flask(__name__)

@app.route('/')
def home():
    """صفحه اصلی"""
    return jsonify({
        "status": "online",
        "service": "Crypto AI Analyst",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/health": "سلامت سیستم",
            "/analysis": "تحلیل بازار (۳ کوین برتر)",
            "/analysis/<symbol>": "تحلیل یک کوین خاص",
            "/system": "وضعیت سیستم"
        }
    })

@app.route('/health')
def health():
    """بررسی سلامت سیستم"""
    try:
        from main_ai import CryptoAIAnalyst
        ai = CryptoAIAnalyst()
        system_status = ai.get_system_status()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": system_status
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analysis')
def market_analysis():
    """تحلیل کلی بازار"""
    try:
        from main_ai import CryptoAIAnalyst
        ai = CryptoAIAnalyst()
        analysis = ai.comprehensive_analysis(top_coins=3)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500

@app.route('/analysis/<symbol>')
def coin_analysis(symbol):
    """تحلیل یک کوین خاص"""
    try:
        from main_ai import CryptoAIAnalyst
        ai = CryptoAIAnalyst()
        ai.load_market_data()
        analysis = ai.generate_trading_strategy(symbol.upper())
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500

@app.route('/system')
def system_info():
    """اطلاعات سیستم"""
    try:
        from main_ai import CryptoAIAnalyst
        ai = CryptoAIAnalyst()
        return jsonify(ai.get_system_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
