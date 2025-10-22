# 📁 src/ai_ml/regime_classifier.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import joblib
import os

class MarketRegimeClassifier:
    """طبقه‌بند رژیم‌های بازار با Random Forest"""
    
    def __init__(self, model_path: str = "models/market_regime_classifier.pkl"):
        self.model_path = model_path
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # تعریف رژیم‌های بازار
        self.regimes = {
            0: 'BULL_NORMAL',      # روند صعودی نرمال
            1: 'BULL_ACCELERATING', # شتاب صعودی
            2: 'BEAR_NORMAL',      # روند نزولی نرمال  
            3: 'BEAR_ACCELERATING', # شتاب نزولی
            4: 'SIDEWAYS_LOW_VOL', # رنج با نوسان کم
            5: 'SIDEWAYS_HIGH_VOL', # رنج با نوسان بالا
            6: 'VOLATILE_BREAKOUT' # نوسان بالا با شکست
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """آماده‌سازی ویژگی‌ها برای مدل"""
        features = pd.DataFrame(index=data.index)
        
        # ویژگی‌های قیمتی
        features['returns_1d'] = data['close'].pct_change(1)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # نوسان
        features['volatility_5d'] = data['close'].pct_change().rolling(5).std()
        features['volatility_20d'] = data['close'].pct_change().rolling(20).std()
        
        # روند
        features['ma_ratio_20_50'] = (
            data['close'].rolling(20).mean() / 
            data['close'].rolling(50).mean() - 1
        )
        features['price_vs_ma20'] = (
            data['close'] / data['close'].rolling(20).mean() - 1
        )
        
        # حجم
        if 'volume' in data.columns:
            features['volume_ratio'] = (
                data['volume'] / data['volume'].rolling(20).mean()
            )
        
        # محدوده معاملاتی
        features['range_5d'] = (
            (data['high'].rolling(5).max() - data['low'].rolling(5).min()) / 
            data['close'].rolling(5).mean()
        )
        
        # مومنتوم
        features['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
        
        return features.dropna()
    
    def create_labels(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """ایجاد لیبل‌های رژیم بازار"""
        labels = pd.Series(index=features.index, dtype=int)
        
        returns_20d = data['close'].pct_change(20)
        volatility_20d = data['close'].pct_change().rolling(20).std()
        
        for i in range(len(features)):
            ret = returns_20d.iloc[i]
            vol = volatility_20d.iloc[i]
            
            if ret > 0.05 and vol < 0.02:
                labels.iloc[i] = 0  # BULL_NORMAL
            elif ret > 0.10 and vol > 0.03:
                labels.iloc[i] = 1  # BULL_ACCELERATING
            elif ret < -0.05 and vol < 0.02:
                labels.iloc[i] = 2  # BEAR_NORMAL
            elif ret < -0.10 and vol > 0.03:
                labels.iloc[i] = 3  # BEAR_ACCELERATING
            elif abs(ret) < 0.02 and vol < 0.015:
                labels.iloc[i] = 4  # SIDEWAYS_LOW_VOL
            elif abs(ret) < 0.03 and vol > 0.025:
                labels.iloc[i] = 5  # SIDEWAYS_HIGH_VOL
            else:
                labels.iloc[i] = 6  # VOLATILE_BREAKOUT
        
        return labels
    
    def train(self, data: pd.DataFrame) -> Dict:
        """آموزش مدل طبقه‌بندی"""
        print("🔄 Training Market Regime Classifier...")
        
        # آماده‌سازی ویژگی‌ها و لیبل‌ها
        features = self.prepare_features(data)
        labels = self.create_labels(data, features)
        
        if len(features) < 100:
            print("❌ Insufficient data for training")
            return {}
        
        # تقسیم داده
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, shuffle=False
        )
        
        # نرمال‌سازی
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # آموزش مدل
        self.model.fit(X_train_scaled, y_train)
        
        # ارزیابی مدل
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        # ذخیره مدل
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': features.columns.tolist()
        }, self.model_path)
        
        print(f"✅ Model trained successfully")
        print(f"   Train Accuracy: {train_score:.3f}")
        print(f"   Test Accuracy: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'training_samples': len(X_train),
            'feature_importance': dict(zip(
                features.columns, 
                self.model.feature_importances_
            ))
        }
    
    def predict_regime(self, data: pd.DataFrame) -> Dict:
        """پیش‌بینی رژیم بازار برای داده جدید"""
        if not self.is_trained:
            # تلاش برای لود مدل ذخیره شده
            if not self.load_model():
                return {'regime': 'UNKNOWN', 'confidence': 0.0}
        
        # آماده‌سازی ویژگی‌ها
        features = self.prepare_features(data)
        if features.empty:
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
        
        # استفاده از آخرین داده
        latest_features = features.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # پیش‌بینی
        prediction = self.model.predict(latest_features_scaled)[0]
        probabilities = self.model.predict_proba(latest_features_scaled)[0]
        
        confidence = probabilities[prediction]
        regime = self.regimes.get(prediction, 'UNKNOWN')
        
        return {
            'regime': regime,
            'confidence': confidence,
            'all_probabilities': {
                self.regimes[i]: prob for i, prob in enumerate(probabilities)
            },
            'features_used': features.columns.tolist()
        }
    
    def load_model(self) -> bool:
        """لود مدل ذخیره شده"""
        try:
            if os.path.exists(self.model_path):
                saved_data = joblib.load(self.model_path)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.is_trained = True
                print("✅ Model loaded successfully")
                return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        
        return False
    
    def get_regime_description(self, regime: str) -> str:
        """توضیحات هر رژیم بازار"""
        descriptions = {
            'BULL_NORMAL': 'روند صعودی پایدار با نوسان متوسط - مناسب برای موقعیت‌های خرید',
            'BULL_ACCELERATING': 'شتاب صعودی با نوسان بالا - فرصت‌های سود سریع اما پرریسک',
            'BEAR_NORMAL': 'روند نزولی پایدار - مناسب برای موقعیت‌های فروش',
            'BEAR_ACCELERATING': 'شتاب نزولی با نوسان بالا - ریسک زیاد، نیاز به مدیریت دقیق',
            'SIDEWAYS_LOW_VOL': 'بازار رنج با نوسان کم - مناسب برای استراتژی‌های رنج',
            'SIDEWAYS_HIGH_VOL': 'بازار رنج با نوسان بالا - فرصت برای نوسان‌گیری',
            'VOLATILE_BREAKOUT': 'نوسان بالا با پتانسیل شکست - نیاز به احتیاط بالا'
        }
        return descriptions.get(regime, 'توضیحی در دسترس نیست')
