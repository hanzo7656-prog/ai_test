# 📁 tests/performance/test_inference_speed.py

import pytest
import time
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.spiking_transformer.transformer_block import SpikingTransformerBlock

class TestInferenceSpeed:
    """تست سرعت inference"""
    
    def setup_method(self):
        self.transformer = SpikingTransformerBlock(
            d_model=64,
            n_heads=4,
            seq_len=10
        )
    
    def test_transformer_inference_speed(self):
        """تست سرعت inference ترانسفورمر"""
        # داده ورودی نمونه
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        test_input = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        # اندازه‌گیری زمان inference
        start_time = time.time()
        
        # اجرای چندین بار برای میانگین‌گیری
        for _ in range(100):
            output = self.transformer.forward(test_input)
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / 100
        
        # میانگین زمان inference باید کمتر از 10ms باشد
        assert avg_inference_time < 0.01  # 10ms
    
    def test_batch_processing_speed(self):
        """تست سرعت پردازش batch"""
        batch_sizes = [1, 2, 4, 8]
        seq_len = 10
        d_model = 64
        
        for batch_size in batch_sizes:
            test_input = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            start_time = time.time()
            output = self.transformer.forward(test_input)
            inference_time = time.time() - start_time
            
            # زمان inference باید با افزایش batch size به صورت خطی افزایش یابد
            if batch_size > 1:
                assert inference_time < batch_size * 0.005  # 5ms per batch
    
    def test_memory_efficiency_during_inference(self):
        """تست کارایی حافظه در حین inference"""
        import psutil
        process = psutil.Process()
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # اجرای inference
        test_input = np.random.randn(4, 10, 64).astype(np.float32)
        for _ in range(50):
            output = self.transformer.forward(test_input)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # مصرف حافظه نباید به شدت افزایش یابد
        memory_increase = memory_after - memory_before
        assert memory_increase < 50  # کمتر از 50MB افزایش
    
    def test_spike_processing_efficiency(self):
        """تست کارایی پردازش اسپایک"""
        spike_stats = self.transformer.get_spike_statistics()
        
        assert 'total_spikes' in spike_stats
        assert 'spike_rates' in spike_stats
        
        # نرخ اسپایک باید معقول باشد
        for rate in spike_stats['spike_rates'].values():
            assert 0 <= rate <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
