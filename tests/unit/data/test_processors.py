# 📁 tests/unit/data/test_processors.py

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data.minimal_processors.feature_selector import FeatureSelector, FeatureConfig
from src.data.minimal_processors.data_compressor import DataCompressor
from src.data.minimal_processors.normalizer import SmartNormalizer
from src.data.processing_pipeline import DataProcessingPipeline

class TestFeatureSelector:
    """تست انتخابگر ویژگی‌ها"""
    
    def setup_method(self):
        self.selector = FeatureSelector()
        self.sample_data = {
            'result': [
                {
                    'id': 'bitcoin',
                    'price': 50000,
                    'volume': 1000000,
                    'marketCap': 1000000000,
                    'priceChange1h': 0.5,
                    'priceChange1d': 2.0,
                    'priceChange1w': 5.0,
                    'availableSupply': 19000000,
                    'totalSupply': 21000000,
                    'rank': 1
                }
            ]
        }
    
    def test_select_features(self):
        """تست انتخاب ویژگی‌ها"""
        result = self.selector.select_features(self.sample_data)
        
        assert result is not None
        assert 'result' in result
        assert len(result['result']) == 1
        assert 'feature_stats' in result
    
    def test_feature_config(self):
        """تست پیکربندی ویژگی‌ها"""
        config = FeatureConfig()
        
        assert len(config.all_features) == 20
        assert 'price' in config.price_features
        assert 'volume' in config.volume_features
    
    def test_extract_critical_features(self):
        """تست استخراج ویژگی‌های حیاتی"""
        coin_data = self.sample_data['result'][0]
        features = self.selector._extract_critical_features(coin_data)
        
        assert len(features) > 10
        assert 'coin_id' in features
        assert 'price_price' in features

class TestDataCompressor:
    """تست فشرده‌ساز داده"""
    
    def setup_method(self):
        self.compressor = DataCompressor()
        self.sample_features = {
            'result': [
                {
                    'price_price': 50000.123456,
                    'volume_volume': 1000000.789,
                    'change_priceChange1h': 0.51234,
                    'coin_id': 'bitcoin'
                }
            ]
        }
    
    def test_compress_features(self):
        """تست فشرده‌سازی ویژگی‌ها"""
        result = self.compressor.compress_features(self.sample_features)
        
        assert result is not None
        assert 'compression_stats' in result
        assert result['compression_stats']['savings_percent'] >= 0
    
    def test_compress_numeric(self):
        """تست فشرده‌سازی اعداد"""
        compressed = self.compressor._compress_numeric(123.456789, 'price_')
        assert compressed == 123.46  # 2 رقم اعشار برای قیمت
        
        compressed = self.compressor._compress_numeric(123.456789, 'volume_')
        assert compressed == 123  # عدد صحیح برای حجم

class TestSmartNormalizer:
    """تست نرمال‌ساز هوشمند"""
    
    def setup_method(self):
        self.normalizer = SmartNormalizer()
        self.sample_data = {
            'result': [
                {'price': 50000, 'volume': 1000000, 'rsi': 45},
                {'price': 51000, 'volume': 1200000, 'rsi': 55},
                {'price': 49000, 'volume': 800000, 'rsi': 35}
            ]
        }
    
    def test_fit_transform(self):
        """تست آموزش و تبدیل"""
        result = self.normalizer.fit_transform(self.sample_data)
        
        assert result is not None
        assert 'result' in result
        assert 'normalization_info' in result
    
    def test_compute_feature_stats(self):
        """تست محاسبه آمار ویژگی‌ها"""
        self.normalizer._compute_feature_stats(self.sample_data['result'])
        
        assert 'price' in self.normalizer.feature_stats
        assert 'mean' in self.normalizer.feature_stats['price']

class TestProcessingPipeline:
    """تست پایپ‌لاین پردازش"""
    
    def setup_method(self):
        self.pipeline = DataProcessingPipeline()
        self.sample_raw_data = {
            'result': [
                {
                    'id': 'bitcoin',
                    'price': 50000,
                    'volume': 1000000,
                    'marketCap': 1000000000,
                    'priceChange1h': 0.5,
                    'priceChange1d': 2.0,
                    'availableSupply': 19000000
                }
            ],
            'meta': {'page': 1, 'limit': 1}
        }
    
    def test_process_raw_data(self):
        """تست پردازش داده خام"""
        result = self.pipeline.process_raw_data(self.sample_raw_data)
        
        assert result is not None
        assert 'pipeline_stats' in self.pipeline.get_pipeline_info()
    
    def test_pipeline_info(self):
        """تست اطلاعات پایپ‌لاین"""
        info = self.pipeline.get_pipeline_info()
        
        assert 'pipeline_stats' in info
        assert 'feature_summary' in info
        assert 'performance_summary' in info

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
