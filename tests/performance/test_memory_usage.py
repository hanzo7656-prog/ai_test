# 📁 tests/performance/test_memory_usage.py

import pytest
import psutil
import os
import gc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.engine import CryptoAnalysisEngine
from src.core.optimization.memory_optimizer import MemoryOptimizer

class TestMemoryUsage:
    """تست مصرف حافظه"""
    
    def setup_method(self):
        self.memory_optimizer = MemoryOptimizer(max_memory_mb=512)
        self.process = psutil.Process()
    
    def test_initial_memory_usage(self):
        """تست مصرف حافظه اولیه"""
        initial_memory = self.memory_optimizer.get_memory_usage()
        
        assert 'rss_mb' in initial_memory
        assert 'vms_mb' in initial_memory
        assert 'percent' in initial_memory
        assert initial_memory['rss_mb'] > 0
    
    def test_memory_cleanup(self):
        """تست پاک‌سازی حافظه"""
        # ایجاد برخی اشیاء برای تست
        test_objects = [list(range(10000)) for _ in range(100)]
        
        memory_before = self.memory_optimizer.get_memory_usage()
        
        # پاک‌سازی
        del test_objects
        gc.collect()
        
        memory_after = self.memory_optimizer.get_memory_usage()
        
        # حافظه باید کاهش یافته باشد
        assert memory_after['rss_mb'] <= memory_before['rss_mb'] * 1.1  # حاشیه خطا
    
    def test_memory_optimization(self):
        """تست بهینه‌سازی حافظه"""
        # شبیه‌سازی یک مدل ساده
        class MockModel:
            def parameters(self):
                return []
        
        model = MockModel()
        self.memory_optimizer.optimize_model_memory(model)
        
        # باید بدون خطا اجرا شود
        assert True
    
    def test_memory_threshold(self):
        """تست آستانه حافظه"""
        should_cleanup = self.memory_optimizer.should_cleanup()
        
        assert isinstance(should_cleanup, bool)
    
    def test_engine_memory_efficiency(self):
        """تست کارایی حافظه موتور"""
        config = {
            'transformer': {'d_model': 64, 'n_heads': 4, 'seq_len': 10},
            'risk': {'total_capital': 10000, 'max_risk_per_trade': 0.02},
            'default_symbols': ['BTC/USDT']
        }
        
        engine = CryptoAnalysisEngine(config)
        
        # مصرف حافظه نباید از 100MB بیشتر شود در مقداردهی اولیه
        memory_usage = self.memory_optimizer.get_memory_usage()
        assert memory_usage['rss_mb'] < 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
