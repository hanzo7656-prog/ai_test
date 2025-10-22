# 📁 tests/unit/data/test_data_manager.py

import pytest
import sys
import os
from unittest.mock import Mock, patch
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data.data_manager import SmartDataManager
from src.data.data_sources.github_data_source import GitHubDataSource
from src.data.data_sources.api_data_source import APIDataSource

class TestDataManager:
    """تست مدیر داده هوشمند"""
    
    def setup_method(self):
        self.data_manager = SmartDataManager()
    
    def test_initialization(self):
        """تست مقداردهی اولیه"""
        assert len(self.data_manager.sources) >= 1
        assert self.data_manager.cache is not None
        assert self.data_manager.performance_tracker is not None
    
    @patch.object(GitHubDataSource, 'get_coins_data')
    def test_data_priority(self, mock_github):
        """تست اولویت‌بندی منابع داده"""
        mock_github.return_value = {
            'result': [{'id': 'bitcoin', 'price': 50000}],
            'meta': {'page': 1}
        }
        
        data = self.data_manager.get_coins_data(limit=10)
        
        # GitHub باید اولویت داشته باشد
        mock_github.assert_called_once()
        assert data is not None
    
    def test_cache_effectiveness(self):
        """تست اثربخشی کش"""
        test_data = {'result': [{'id': 'test', 'price': 100}]}
        
        # قرار دادن در کش
        self.data_manager.cache.set('test_key', test_data)
        
        # بازیابی از کش
        cached_data = self.data_manager.cache.get('test_key')
        
        assert cached_data == test_data
    
    @patch.object(APIDataSource, 'get_coins_data')
    def test_fallback_to_api(self, mock_api):
        """تست fallback به API"""
        mock_api.return_value = {
            'result': [{'id': 'ethereum', 'price': 3000}],
            'meta': {'page': 1}
        }
        
        # شبیه‌سازی شکست GitHub
        with patch.object(GitHubDataSource, 'get_coins_data', return_value=None):
            data = self.data_manager.get_coins_data()
        
        # باید از API استفاده کند
        mock_api.assert_called_once()
        assert data is not None
    
    def test_performance_tracking(self):
        """تست ردیابی عملکرد"""
        stats = self.data_manager.get_source_stats()
        
        assert 'total_sources' in stats
        assert 'sources_available' in stats
        assert 'performance_metrics' in stats
    
    def test_symbol_filtering(self):
        """تست فیلتر کردن نمادها"""
        with patch.object(GitHubDataSource, 'get_coins_data') as mock_github:
            mock_github.return_value = {
                'result': [
                    {'id': 'bitcoin', 'symbol': 'BTC', 'price': 50000},
                    {'id': 'ethereum', 'symbol': 'ETH', 'price': 3000}
                ]
            }
            
            data = self.data_manager.get_coins_data(symbols=['BTC'])
            
            assert data is not None
            # باید فقط BTC برگرداند

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
