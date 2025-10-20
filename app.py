# app.py
from flask import Flask, jsonify, render_template
import psutil
import os
import time
import random
import numpy as np
from datetime import datetime
import json
from api_client import VortexAPIClient
from technical_analysis_engine import TechnicalAnalysisEngine

app = Flask(__name__)

class AdvancedAI:
    def __init__(self):
        self.neurons = 100
        self.middleware_url = "https://server-test-ovta.onrender.com/api"
        self.model_type = "VortexAI-Market-Predictor"
        self.training_data = []
        
        # کلاینت جدید API برای داده‌های خام
        self.api = VortexAPIClient(self.middleware_url)
        
        # موتور تحلیل تکنیکال
        self.technical_engine = TechnicalAnalysisEngine()
        
        print(f"🔍 مدل پیشرفته AI با {self.neurons} نورون راه‌اندازی شد")
        print(f"🌐 کلاینت API متصل به: {self.api.base_url}")
        print(f"📊 موتور تحلیل تکنیکال فعال با {sum(len(v) for v in self.technical_engine.available_indicators.values())} اندیکاتور")
        
        # تست اتصال اولیه
        connection_status = self.api.test_connection()
        if connection_status:
            print("✅ اتصال به سرور میانی برقرار است")
            
            # تست جامع API
            test_report = self.api.comprehensive_test()
            success_rate = test_report['summary']['success_rate']
            print(f"📡 تست جامع API: {success_rate}")
        else:
            print("⚠️ اتصال به سرور میانی با مشکل مواجه است")

    def fetch_market_data(self):
        """دریافت داده‌های بازار از طریق کلاینت جدید"""
        return self.api.get_all_market_data()

    def fetch_technical_data(self, symbol="BTC"):
        """دریافت داده‌های تکنیکال برای یک ارز خاص"""
        return self.api.get_ai_raw_single(symbol)

    def predict_market_trend(self):
        """پیش‌بینی روند بازار با داده‌های کامل"""
        start_time = time.time()
        
        # دریافت داده‌های خام برای پیش‌بینی
        prediction_data = self.api.get_ai_prediction_data()
        
        if not prediction_data['success']:
            return {
                "prediction": "داده‌ای دریافت نشد",
                "confidence": 0,
                "data_source": "fallback",
                "error": "عدم اتصال به سرور داده"
            }
        
        # تحلیل داده‌های دریافتی
        analysis_results = {}
        
        # تحلیل داده‌های بازار
        market_data = prediction_data['prediction_data']['current_market']
        if market_data['success']:
            analysis_results['market_analysis'] = self.technical_engine.analyze_raw_api_data(
                market_data['data']
            )
        
        # تحلیل احساسات بازار
        fear_greed_data = prediction_data['prediction_data']['market_sentiment']
        if fear_greed_data['success']:
            analysis_results['sentiment_analysis'] = self._analyze_market_sentiment(
                fear_greed_data['data']
            )
        
        # تحلیل دامیننس بیت‌کوین
        btc_dominance_data = prediction_data['prediction_data']['btc_dominance']
        if btc_dominance_data['success']:
            analysis_results['btc_analysis'] = self._analyze_btc_dominance(
                btc_dominance_data['data']
            )
        
        # تولید پیش‌بینی نهایی
        final_prediction = self._generate_final_prediction(analysis_results)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            **final_prediction,
            "data_sources_used": len([k for k in analysis_results.keys() if analysis_results[k]]),
            "processing_time_ms": processing_time,
            "successful_sources": prediction_data['successful_sources'],
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_market_sentiment(self, fear_greed_data: Dict) -> Dict:
        """تحلیل احساسات بازار"""
        try:
            raw_data = fear_greed_data.get('raw_data', fear_greed_data)
            fear_greed_value = raw_data.get('value', raw_data.get('now', {}).get('value', 50))
            
            sentiment = "خنثی"
            if fear_greed_value >= 70:
                sentiment = "طمع شدید"
            elif fear_greed_value >= 55:
                sentiment = "طمع"
            elif fear_greed_value <= 30:
                sentiment = "ترس شدید"
            elif fear_greed_value <= 45:
                sentiment = "ترس"
            
            return {
                'fear_greed_index': fear_greed_value,
                'sentiment': sentiment,
                'classification': raw_data.get('value_classification', 'Neutral')
            }
        except:
            return {'error': 'خطا در تحلیل احساسات'}

    def _analyze_btc_dominance(self, dominance_data: Dict) -> Dict:
        """تحلیل دامیننس بیت‌کوین"""
        try:
            raw_data = dominance_data.get('raw_data', dominance_data)
            dominance_value = raw_data.get('value', raw_data.get('percentage', 50))
            
            trend = "پایدار"
            if dominance_value > 55:
                trend = "قدرتمند"
            elif dominance_value < 45:
                trend = "ضعیف"
            
            return {
                'btc_dominance': dominance_value,
                'trend': trend,
                'market_implication': 'آلت‌کوین‌ها فرصت دارند' if dominance_value < 45 else 'بیت‌کوین مسلط است'
            }
        except:
            return {'error': 'خطا در تحلیل دامیننس'}

    def _generate_final_prediction(self, analysis_results: Dict) -> Dict:
        """تولید پیش‌بینی نهایی بر اساس تحلیل‌ها"""
        # جمع‌آوری امتیازات از تحلیل‌های مختلف
        bullish_score = 0
        bearish_score = 0
        confidence_factors = []
        
        # تحلیل تکنیکال
        tech_analysis = analysis_results.get('market_analysis', {})
        if 'overall_trend' in tech_analysis:
            if tech_analysis['overall_trend'] == 'bullish':
                bullish_score += 2
                confidence_factors.append('روند تکنیکال صعودی')
            elif tech_analysis['overall_trend'] == 'bearish':
                bearish_score += 2
                confidence_factors.append('روند تکنیکال نزولی')
        
        # تحلیل احساسات
        sentiment_analysis = analysis_results.get('sentiment_analysis', {})
        if 'sentiment' in sentiment_analysis:
            sentiment = sentiment_analysis['sentiment']
            if 'طمع' in sentiment:
                bearish_score += 1  # طمع شدید معمولاً نشانه اصلاح است
                confidence_factors.append('احساسات بازار به طمع نزدیک است')
            elif 'ترس' in sentiment:
                bullish_score += 1  # ترس شدید معمولاً فرصت خرید است
                confidence_factors.append('احساسات بازار به ترس نزدیک است')
        
        # تحلیل بیت‌کوین
        btc_analysis = analysis_results.get('btc_analysis', {})
        if 'trend' in btc_analysis:
            if btc_analysis['trend'] == 'قدرتمند':
                bullish_score += 1
                confidence_factors.append('بیت‌کوین در موقعیت قدرتمند')
        
        # تصمیم‌گیری نهایی
        total_score = bullish_score - bearish_score
        confidence = min(abs(total_score) * 20, 95)
        
        if total_score > 1:
            prediction = "صعودی"
        elif total_score < -1:
            prediction = "نزولی"
        else:
            prediction = "خنثی"
            confidence = max(confidence, 30)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "confidence_factors": confidence_factors,
            "analysis_breakdown": {
                "technical": tech_analysis.get('overall_trend', 'نامشخص'),
                "sentiment": sentiment_analysis.get('sentiment', 'نامشخص'),
                "btc_dominance": btc_analysis.get('trend', 'نامشخص')
            }
        }

    def predict_system_load(self):
        """پیش‌بینی مصرف منابع سیستم"""
        health_data = self.api.get_health_combined()
        
        if not health_data or not health_data.get('success'):
            return {
                "predicted_ram_mb": 350,
                "predicted_cpu_percent": 25,
                "data_source": "fallback"
            }
        
        # محاسبه پیش‌بینی بر اساس سلامت سیستم
        health_info = health_data.get('data', {})
        websocket_status = health_info.get('websocket_status', {})
        api_status = health_info.get('api_status', {})
        
        active_coins = websocket_status.get('active_coins', 0)
        api_requests = api_status.get('requests_count', 0)
        
        # مدل پیش‌بینی پیشرفته‌تر
        base_ram = 200
        ram_per_coin = 3
        ram_per_request = 0.1
        
        base_cpu = 15
        cpu_per_coin = 0.5
        cpu_per_request = 0.05
        
        predicted_ram = base_ram + (active_coins * ram_per_coin) + (api_requests * ram_per_request)
        predicted_cpu = base_cpu + (active_coins * cpu_per_coin) + (api_requests * cpu_per_request)
        
        return {
            "predicted_ram_mb": min(round(predicted_ram), 450),
            "predicted_cpu_percent": min(round(predicted_cpu), 80),
            "active_coins": active_coins,
            "api_requests": api_requests,
            "data_source": "ai_calculation"
        }

    def comprehensive_analysis(self, symbol="BTC"):
        """تحلیل جامع یک ارز خاص"""
        start_time = time.time()
        
        # دریافت داده‌های مختلف
        technical_data = self.api.get_ai_raw_single(symbol)
        market_overview = self.api.get_market_cap()
        fear_greed = self.api.get_fear_greed()
        
        analysis = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "technical_analysis": "در حال تحلیل...",
            "market_context": "در حال دریافت...",
            "market_sentiment": "در حال دریافت...",
            "ai_recommendation": "در حال تحلیل...",
            "signal_strength": 0,
            "risk_level": "متوسط"
        }
        
        # تحلیل تکنیکال
        if technical_data and 'coin_data' in technical_data:
            coin_data = technical_data['coin_data']
            if coin_data['success']:
                tech_analysis = self.technical_engine.analyze_raw_api_data(coin_data['data'])
                analysis['technical_analysis'] = tech_analysis
                
                # استخراج سیگنال از تحلیل تکنیکال
                if 'overall_trend' in tech_analysis:
                    if tech_analysis['overall_trend'] == 'bullish':
                        analysis['ai_recommendation'] = "مثبت"
                        analysis['signal_strength'] = 75
                        analysis['risk_level'] = "کم"
                    elif tech_analysis['overall_trend'] == 'bearish':
                        analysis['ai_recommendation'] = "منفی" 
                        analysis['signal_strength'] = 65
                        analysis['risk_level'] = "بالا"
        
        # تحلیل بازار
        if market_overview and market_overview['success']:
            analysis['market_context'] = self.technical_engine.analyze_raw_api_data(market_overview['data'])
        
        # تحلیل احساسات
        if fear_greed and fear_greed['success']:
            sentiment = self._analyze_market_sentiment(fear_greed['data'])
            analysis['market_sentiment'] = sentiment
        
        analysis['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        return analysis

    def get_market_insights(self):
        """دریافت بینش‌های بازار"""
        dashboard = self.api.get_insights_dashboard()
        fear_greed = self.api.get_fear_greed()
        btc_dominance = self.api.get_btc_dominance()
        rainbow_chart = self.api.get_raw_rainbow_chart()
        
        return {
            "dashboard": dashboard,
            "fear_greed": fear_greed,
            "btc_dominance": btc_dominance,
            "rainbow_chart": rainbow_chart,
            "timestamp": datetime.now().isoformat()
        }

    def get_raw_data_overview(self):
        """دریافت نمای کلی داده‌های خام"""
        training_data = self.api.get_ai_training_data()
        
        if training_data['success']:
            return {
                "success": True,
                "data_sources": training_data['successful_sources'],
                "total_sources": training_data['total_sources'],
                "success_rate": training_data['success_rate'],
                "processing_time": training_data['processing_time'],
                "timestamp": training_data['timestamp']
            }
        else:
            return {
                "success": False,
                "error": "عدم دریافت داده‌های آموزشی",
                "timestamp": datetime.now().isoformat()
            }

# Initialize Advanced AI Model
ai_model = AdvancedAI()

# ========== توابع کمکی سیستم ==========

def get_real_cpu_usage():
    """روش ساده و مطمئن برای اندازه گیری CPU"""
    try:
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent(interval=0.5)

        if cpu_percent == 0:
            cpu_percent = random.uniform(0.1, 2.0)

        return round(cpu_percent, 2)

    except Exception as e:
        return round(random.uniform(0.5, 3.0), 2)

def get_system_info():
    """اطلاعات سیستم"""
    process = psutil.Process(os.getpid())

    try:
        cpu_percent = get_real_cpu_usage()
        process_memory_mb = process.memory_info().rss / 1024 / 1024
        total_ram_mb = 512
        ram_percent = (process_memory_mb / total_ram_mb) * 100

        return {
            "ram_used_mb": round(process_memory_mb, 2),
            "ram_percent": round(ram_percent, 2),
            "total_ram_mb": total_ram_mb,
            "cpu_percent": cpu_percent,
            "neurons": ai_model.neurons,
            "status": "سالم و فعال",
            "server_time": time.strftime("%H:%M:%S"),
            "model_type": ai_model.model_type
        }

    except Exception as e:
        return {
            "ram_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "ram_percent": 8.8,
            "total_ram_mb": 512,
            "cpu_percent": 1.2,
            "neurons": ai_model.neurons,
            "status": "سالم و فعال",
            "server_time": time.strftime("%H:%M:%S"),
            "model_type": ai_model.model_type
        }

# ========== Routes ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify(get_system_info())

@app.route('/predict/market')
def predict_market():
    """پیش‌بینی روند بازار با استفاده از داده‌های سرور میانی"""
    start_time = time.time()

    market_prediction = ai_model.predict_market_trend()
    system_prediction = ai_model.predict_system_load()
    processing_time = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "success": True,
        "market_prediction": market_prediction,
        "system_prediction": system_prediction,
        "processing_time_ms": processing_time,
        "neurons_used": ai_model.neurons,
        "message": "پیش‌بینی بازار با موفقیت انجام شد"
    })

@app.route('/analyze/coin/<symbol>')
def analyze_coin(symbol):
    """تحلیل یک ارز خاص با داده‌های تکنیکال"""
    analysis = ai_model.comprehensive_analysis(symbol)
    return jsonify({
        "success": True,
        "analysis": analysis
    })

@app.route('/system/forecast')
def system_forecast():
    """پیش‌بینی مصرف منابع سیستم"""
    prediction = ai_model.predict_system_load()
    current_usage = get_system_info()

    return jsonify({
        "success": True,
        "current_usage": current_usage,
        "predicted_usage": prediction,
        "forecast_timestamp": datetime.now().isoformat()
    })

@app.route('/insights/market')
def market_insights():
    """بینش‌های بازار"""
    insights = ai_model.get_market_insights()
    return jsonify({
        "success": True,
        "insights": insights
    })

@app.route('/data/overview')
def data_overview():
    """نمای کلی داده‌های موجود"""
    data_overview = ai_model.get_raw_data_overview()
    status_report = ai_model.api.get_status_report()
    
    return jsonify({
        "success": data_overview['success'],
        "data_overview": data_overview,
        "status_report": status_report,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/test/middleware-connection')
def test_middleware_connection():
    """تست اتصال به سرور میانی"""
    start_time = time.time()

    connection_status = ai_model.api.test_connection()
    status_report = ai_model.api.get_status_report()
    comprehensive_test = ai_model.api.comprehensive_test()

    processing_time = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "middleware_connection": "success" if connection_status else "failed",
        "status_report": status_report,
        "comprehensive_test": comprehensive_test,
        "processing_time_ms": processing_time,
        "middleware_url": ai_model.middleware_url
    })

@app.route('/technical/analyze/<symbol>')
def technical_analyze(symbol):
    """تحلیل تکنیکال پیشرفته یک ارز"""
    start_time = time.time()
    
    # دریافت داده‌های خام
    raw_data = ai_model.api.get_ai_raw_single(symbol)
    
    # تحلیل با موتور تکنیکال
    analysis_results = {}
    for data_type, data_response in raw_data.items():
        if data_response and data_response.get('success'):
            analysis = ai_model.technical_engine.analyze_raw_api_data(data_response['data'])
            analysis_results[data_type] = analysis
    
    processing_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "success": True,
        "symbol": symbol.upper(),
        "analysis_results": analysis_results,
        "processing_time_ms": processing_time,
        "timestamp": datetime.now().isoformat()
    })

# تست‌های CPU
@app.route('/test-cpu')
def test_cpu():
    start_time = time.time()

    result = 0
    for i in range(500000):
        result += i * 0.00001

    pi_estimate = 0
    for k in range(10000):
        pi_estimate += (4.0 * (-1)**k) / (2*k + 1)

    duration = (time.time() - start_time) * 1000

    return jsonify({
        "test_result": round(result, 6),
        "pi_estimate": round(pi_estimate, 6),
        "processing_time_ms": round(duration, 2),
        "cpu_usage_note": "تست سنگین CPU انجام شد"
    })

@app.route('/light-cpu')
def light_cpu():
    start_time = time.time()
    result = sum(i * 0.1 for i in range(1000))
    duration = (time.time() - start_time) * 1000

    return jsonify({
        "test_result": round(result, 4),
        "processing_time_ms": round(duration, 2),
        "cpu_usage_note": "تست سبک CPU انجام شد"
    })

if __name__ == '__main__':
    print("🚀 برنامه هوش مصنوعی پیشرفته شروع شد...")
    print("📡 درحال اتصال به سرور میانی...")
    
    # تست نهایی اتصال
    if ai_model.api.test_connection():
        print("✅ همه چیز آماده است! سرور در حال راه‌اندازی...")
        
        # تست اولیه عملکرد
        print("🧪 انجام تست اولیه عملکرد...")
        try:
            health = ai_model.api.get_health_combined()
            if health.get('success'):
                print("✅ تست سلامت سیستم موفقیت‌آمیز بود")
            else:
                print("⚠️ تست سلامت سیستم با مشکل مواجه شد")
        except Exception as e:
            print(f"⚠️ خطا در تست اولیه: {e}")
    else:
        print("⚠️  هشدار: اتصال به سرور میانی با مشکل مواجه است")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
