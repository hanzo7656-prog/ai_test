# 📁 tests/unit/ai_ml/test_regime_classifier.py

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.ai_ml.regime_classifier import MarketRegimeClassifier

class TestRegimeClassifier:
    """تست طبقه‌بند رژیم بازار"""
    
    def setup_method(self):
        self.classifier = MarketRegimeClassifier()
        
        # داده نمونه برای تست
        dates = pd.date_range('2023-01-01', periods=100, freq='1d')
        prices = 50000 + np.cumsum(np.random.randn(100) * 1000)
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 50,
            'high': prices + np.abs(np.random.randn(100) * 100),
            'low': prices - np.abs(np.random.randn(100) * 100),
            'close': prices,
            'volume': np.random.randint(10000, 100000, 100)
        }, index=dates)
    
    def test_feature_preparation(self):
        """تست آماده‌سازی ویژگی‌ها"""
        features = self.classifier.prepare_features(self.sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'returns_1d' in features.columns
        assert 'volatility_5d' in features.columns
    
    def test_label_creation(self):
        """تست ایجاد لیبل‌ها"""
        features = self.classifier.prepare_features(self.sample_data)
        labels = self.classifier.create_labels(self.sample_data, features)
        
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(features)
        assert all(label in range(7) for label in labels.unique())
    
    def test_model_training(self):
        """تست آموزش مدل"""
        result = self.classifier.train(self.sample_data)
        
        # در صورت داده کافی باید آموزش موفق باشد
        if len(self.sample_data) >= 100:
            assert 'train_accuracy' in result
            assert 'test_accuracy' in result
            assert 'feature_importance' in result
        else:
            assert result == {}  # داده ناکافی
    
    def test_regime_prediction(self):
        """تست پیش‌بینی رژیم"""
        # آموزش مدل با داده نمونه
        if len(self.sample_data) >= 100:
            self.classifier.train(self.sample_data)
            
            prediction = self.classifier.predict_regime(self.sample_data)
            
            assert 'regime' in prediction
            assert 'confidence' in prediction
            assert 'all_probabilities' in prediction
            assert prediction['confidence'] >= 0
            assert prediction['confidence'] <= 1
    
    def test_regime_descriptions(self):
        """تست توضیحات رژیم‌ها"""
        for regime_id, regime_name in self.classifier.regimes.items():
            description = self.classifier.get_regime_description(regime_name)
            assert isinstance(description, str)
            assert len(description) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
