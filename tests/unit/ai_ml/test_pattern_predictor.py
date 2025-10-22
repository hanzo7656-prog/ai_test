# 📁 tests/unit/ai_ml/test_pattern_predictor.py

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.ai_ml.pattern_predictor import PatternPredictor

class TestPatternPredictor:
    """تست پیش‌بین الگوهای قیمتی"""
    
    def setup_method(self):
        self.predictor = PatternPredictor(sequence_length=20)
        
        # داده نمونه برای تست
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.randn(200) * 10,
            'high': prices + np.abs(np.random.randn(200) * 20),
            'low': prices - np.abs(np.random.randn(200) * 20),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
    
    def test_feature_extraction(self):
        """تست استخراج ویژگی‌ها"""
        features = self.predictor._extract_features(self.sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.sample_data)
        assert 'open' in features.columns
        assert 'close' in features.columns
        assert 'volume' in features.columns
        assert 'returns' in features.columns
    
    def test_sequence_preparation(self):
        """تست آماده‌سازی دنباله‌ها"""
        X, y = self.predictor.prepare_sequences(self.sample_data)
        
        if len(self.sample_data) > self.predictor.sequence_length:
            assert len(X) > 0
            assert len(y) > 0
            assert X.shape[1] == self.predictor.sequence_length
            assert all(label in [0, 1, 2] for label in y)
    
    def test_model_initialization(self):
        """تست مقداردهی اولیه مدل"""
        assert self.predictor.model is None
        assert not self.predictor.is_trained
        assert self.predictor.scaler is not None
    
    def test_pattern_prediction(self):
        """تست پیش‌بینی الگو"""
        # آموزش با داده محدود برای تست سریع
        if len(self.sample_data) > 50:
            result = self.predictor.train(self.sample_data.iloc[:50], epochs=10)
            
            if result:  # اگر آموزش موفق بود
                prediction = self.predictor.predict_pattern(self.sample_data)
                
                assert 'pattern' in prediction
                assert 'confidence' in prediction
                assert 'all_probabilities' in prediction
                assert prediction['confidence'] >= 0
                assert prediction['confidence'] <= 1
    
    def test_pattern_types(self):
        """تست انواع الگوها"""
        patterns = self.predictor.patterns
        assert len(patterns) == 3
        assert 0 in patterns  # UPTREND_CONTINUATION
        assert 1 in patterns  # DOWNTREND_CONTINUATION  
        assert 2 in patterns  # REVERSAL

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
