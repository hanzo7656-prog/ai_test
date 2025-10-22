# 📁 src/ai_ml/signal_optimizer.py

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SignalOptimizer:
    """بهینه‌ساز سیگنال‌های معاملاتی با Reinforcement Learning"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.is_trained = False
        self.feature_importance = {}
    
    def prepare_training_data(self, historical_data: pd.DataFrame, 
                            signals: List[Dict], actual_returns: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """آماده‌سازی داده برای آموزش"""
        X = []
        y = []
        
        for i, signal in enumerate(signals):
            if i >= len(actual_returns):
                break
            
            # ویژگی‌های سیگنال
            features = [
                signal.get('confidence', 0),
                signal.get('rsi', 50),
                signal.get('volume_ratio', 1),
                signal.get('trend_strength', 0),
                signal.get('volatility', 0.01),
                signal.get('risk_reward_ratio', 1)
            ]
            
            X.append(features)
            y.append(actual_returns[i])
        
        return np.array(X), np.array(y)
    
    def train(self, historical_data: pd.DataFrame, signals: List[Dict], 
              actual_returns: List[float]) -> Dict:
        """آموزش مدل بهینه‌سازی سیگنال"""
        try:
            X, y = self.prepare_training_data(historical_data, signals, actual_returns)
            
            if len(X) < 50:
                logger.warning("Insufficient data for signal optimization training")
                return {}
            
            self.model.fit(X, y)
            self.is_trained = True
            
            # ذخیره اهمیت ویژگی‌ها
            feature_names = ['confidence', 'rsi', 'volume_ratio', 'trend_strength', 'volatility', 'risk_reward']
            self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            # ارزیابی مدل
            train_score = self.model.score(X, y)
            
            logger.info(f"✅ Signal optimizer trained. R² score: {train_score:.3f}")
            
            return {
                'training_samples': len(X),
                'r2_score': train_score,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Signal optimization training failed: {e}")
            return {}
    
    def optimize_signal(self, signal: Dict, market_conditions: Dict) -> Dict:
        """بهینه‌سازی یک سیگنال"""
        if not self.is_trained:
            return signal
        
        try:
            # ویژگی‌های سیگنال
            features = np.array([[
                signal.get('confidence', 0),
                market_conditions.get('rsi', 50),
                market_conditions.get('volume_ratio', 1),
                market_conditions.get('trend_strength', 0),
                market_conditions.get('volatility', 0.01),
                signal.get('risk_reward_ratio', 1)
            ]])
            
            # پیش‌بینی بازدهی بهینه
            predicted_return = self.model.predict(features)[0]
            
            # تنظیم اعتماد بر اساس پیش‌بینی
            optimized_confidence = min(1.0, max(0.0, signal.get('confidence', 0) * (1 + predicted_return)))
            
            # ایجاد سیگنال بهینه‌شده
            optimized_signal = signal.copy()
            optimized_signal['confidence'] = optimized_confidence
            optimized_signal['optimized'] = True
            optimized_signal['predicted_return'] = predicted_return
            optimized_signal['optimization_score'] = self._calculate_optimization_score(
                optimized_confidence, predicted_return
            )
            
            return optimized_signal
            
        except Exception as e:
            logger.error(f"Signal optimization failed: {e}")
            return signal
    
    def _calculate_optimization_score(self, confidence: float, predicted_return: float) -> float:
        """محاسبه نمره بهینه‌سازی"""
        return (confidence * 0.6) + (predicted_return * 0.4)
    
    def get_optimization_metrics(self) -> Dict:
        """دریافت معیارهای بهینه‌سازی"""
        return {
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'model_type': 'GradientBoostingRegressor'
        }
