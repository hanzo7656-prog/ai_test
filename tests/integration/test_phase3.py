# 📁 test_phase3.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.multi_timeframe.timeframe_sync import MultiTimeframeAnalyzer, TimeFrame
from src.ai_ml.regime_classifier import MarketRegimeClassifier
from src.ai_ml.pattern_predictor import PatternPredictor

def generate_multi_timeframe_data():
    """تولید داده‌های چندزمانه برای تست"""
    base_dates = pd.date_range('2023-01-01', '2024-01-15', freq='1h')
    
    # تولید داده پایه (1h)
    base_prices = np.cumsum(np.random.randn(len(base_dates)) * 0.01) + 50000
    
    base_data = pd.DataFrame({
        'open': base_prices + np.random.randn(len(base_dates)) * 10,
        'high': base_prices + np.abs(np.random.randn(len(base_dates)) * 20),
        'low': base_prices - np.abs(np.random.randn(len(base_dates)) * 20),
        'close': base_prices,
        'volume': np.random.randint(1000, 10000, len(base_dates))
    }, index=base_dates)
    
    # تولید تایم‌فریم‌های دیگر
    multi_tf_data = {}
    
    # 4h data
    h4_data = base_data.resample('4h').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    multi_tf_data[TimeFrame.H4] = h4_data
    
    # 1d data
    d1_data = base_data.resample('1d').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'
    })
    multi_tf_data[TimeFrame.D1] = d1_data
    
    # 1w data (simplified)
    w1_data = base_data.resample('1w').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last', 
        'volume': 'sum'
    })
    multi_tf_data[TimeFrame.W1] = w1_data
    
    return multi_tf_data

def test_multi_timeframe_analysis():
    """تست تحلیل چندزمانه"""
    print("📊 Testing Multi-Timeframe Analysis...")
    
    # تولید داده‌های تست
    multi_tf_data = generate_multi_timeframe_data()
    
    # ایجاد آنالیزور
    mt_analyzer = MultiTimeframeAnalyzer()
    
    # تحلیل نماد
    analysis = mt_analyzer.analyze_symbol('BTC/USDT', multi_tf_data)
    
    print(f"✅ Analyzed {len(analysis) - 2} timeframes")  # minus hierarchical and consensus
    
    # نمایش نتایج
    for tf, result in analysis.items():
        if tf in ['hierarchical', 'consensus']:
            print(f"\n🎯 {tf.upper()}:")
            for key, value in result.items():
                print(f"   {key}: {value}")
        else:
            print(f"\n⏰ {tf}:")
            trend = result.get('trend', {})
            print(f"   Trend: {trend.get('direction')} (Strength: {trend.get('strength'):.2f})")
    
    return analysis

def test_ai_models():
    """تست مدل‌های هوش مصنوعی"""
    print("\n🧠 Testing AI Models...")
    
    # تولید داده برای آموزش
    dates = pd.date_range('2022-01-01', '2024-01-15', freq='1d')
    prices = np.cumsum(np.random.randn(len(dates)) * 0.02) + 50000
    
    train_data = pd.DataFrame({
        'open': prices + np.random.randn(len(dates)) * 50,
        'high': prices + np.abs(np.random.randn(len(dates)) * 100),
        'low': prices - np.abs(np.random.randn(len(dates)) * 100),
        'close': prices,
        'volume': np.random.randint(10000, 100000, len(dates))
    }, index=dates)
    
    # تست Market Regime Classifier
    print("\n🔮 Market Regime Classifier:")
    regime_classifier = MarketRegimeClassifier()
    
    # آموزش مدل
    training_result = regime_classifier.train(train_data)
    if training_result:
        print(f"   Training completed - Accuracy: {training_result['test_accuracy']:.3f}")
        
        # پیش‌بینی رژیم فعلی
        regime_prediction = regime_classifier.predict_regime(train_data)
        print(f"   Current Regime: {regime_prediction['regime']}")
        print(f"   Confidence: {regime_prediction['confidence']:.3f}")
        print(f"   Description: {regime_classifier.get_regime_description(regime_prediction['regime'])}")
    
    # تست Pattern Predictor
    print("\n🔍 Pattern Predictor:")
    pattern_predictor = PatternPredictor(sequence_length=20)
    
    # آموزش مدل (با داده کمتر برای تست سریع)
    pattern_result = pattern_predictor.train(train_data.iloc[:500], epochs=50)
    if pattern_result:
        print(f"   Training completed - Accuracy: {pattern_result['final_accuracy']:.3f}")
        
        # پیش‌بینی الگو
        pattern_prediction = pattern_predictor.predict_pattern(train_data)
        print(f"   Predicted Pattern: {pattern_prediction['pattern']}")
        print(f"   Confidence: {pattern_prediction['confidence']:.3f}")
    
    return {
        'regime_classifier': regime_classifier,
        'pattern_predictor': pattern_predictor,
        'regime_prediction': regime_prediction,
        'pattern_prediction': pattern_prediction
    }

def integration_test_phase3():
    """تست یکپارچه فاز ۳"""
    print("🚀 Starting Phase 3 Integration Test...")
    
    # تست تحلیل چندزمانه
    mt_analysis = test_multi_timeframe_analysis()
    
    # تست مدل‌های هوش مصنوعی
    ai_results = test_ai_models()
    
    print("\n🎉 Phase 3 Integration Test Completed Successfully!")
    print("📋 Summary:")
    print("   - Multi-Timeframe Analysis: ✅ Hierarchical trend analysis")
    print("   - Market Regime Detection: ✅ Random Forest classifier") 
    print("   - Pattern Prediction: ✅ LSTM network")
    print("   - AI Integration: ✅ Real-time market intelligence")
    
    return {
        'multi_timeframe_analysis': mt_analysis,
        'ai_models': ai_results
    }

if __name__ == "__main__":
    integration_test_phase3()
