# 📁 src/ai_ml/anomaly_detector.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """تشخیص‌دهنده ناهنجاری در داده‌های بازار"""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """استخراج ویژگی‌ها برای تشخیص ناهنجاری"""
        features = []
        
        # ویژگی‌های قیمتی
        features.append(data['close'].pct_change().rolling(5).std())  # نوسان کوتاه‌مدت
        features.append(data['close'].pct_change().rolling(20).std()) # نوسان بلندمدت
        
        # ویژگی‌های حجم
        if 'volume' in data.columns:
            volume_anomaly = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
            features.append(volume_anomaly)
        
        # ویژگی‌های محدوده قیمت
        price_range = (data['high'] - data['low']) / data['close']
        features.append(price_range)
        
        # ترکیب ویژگی‌ها
        feature_matrix = np.column_stack([f.dropna() for f in features])
        return feature_matrix
    
    def train(self, data: pd.DataFrame) -> Dict:
        """آموزش مدل تشخیص ناهنجاری"""
        try:
            features = self.extract_features(data)
            
            if len(features) < 50:
                logger.warning("Insufficient data for anomaly detection training")
                return {}
            
            # نرمال‌سازی ویژگی‌ها
            features_scaled = self.scaler.fit_transform(features)
            
            # آموزش مدل
            self.model.fit(features_scaled)
            self.is_trained = True
            
            # پیش‌بینی روی داده آموزش
            predictions = self.model.predict(features_scaled)
            anomaly_count = (predictions == -1).sum()
            
            logger.info(f"✅ Anomaly detector trained. Found {anomaly_count} anomalies in training data")
            
            return {
                'training_samples': len(features),
                'anomalies_detected': anomaly_count,
                'anomaly_rate': anomaly_count / len(features)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection training failed: {e}")
            return {}
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict:
        """تشخیص ناهنجاری در داده جدید"""
        if not self.is_trained:
            logger.warning("Anomaly detector not trained")
            return {}
        
        try:
            features = self.extract_features(data)
            
            if len(features) == 0:
                return {}
            
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            anomaly_scores = self.model.decision_function(features_scaled)
            
            # یافتن ناهنجاری‌ها
            anomaly_indices = np.where(predictions == -1)[0]
            anomalies = []
            
            for idx in anomaly_indices:
                anomalies.append({
                    'timestamp': data.index[idx] if hasattr(data, 'index') else idx,
                    'score': float(anomaly_scores[idx]),
                    'features': features[idx].tolist()
                })
            
            return {
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / len(features),
                'anomalies': anomalies,
                'average_anomaly_score': np.mean(anomaly_scores[anomaly_indices]) if len(anomaly_indices) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {}
