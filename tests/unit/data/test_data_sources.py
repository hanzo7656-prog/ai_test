# 📁 tests/unit/data/test_data_sources.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data.data_sources.github_data_source import GitHubDataSource
from src.data.data_sources.api_data_source import APIDataSource
from src.data.data_manager import SmartDataManager

class TestGitHubDataSource:
    """تست منبع داده GitHub"""
    
    def setup_method(self):
        self.github_source = GitHubDataSource()
    
    @patch('requests.get')
    def test_fetch_from_github_success(self, mock_get):
        """تست دریافت موفق از GitHub"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result': [{'id': 'bitcoin', 'price': 50000, 'symbol': 'BTC'}]
        }
        mock_get.return_value = mock_response
        
        data = self.github_source._fetch_from_github('coins')
        
        assert data is not None
        assert 'result' in data
        assert len(data['result']) == 1
    
    @patch('requests.get')
    def test_fetch_from_github_failure(self, mock_get):
        """تست شکست در دریافت از GitHub"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        data = self.github_source._fetch_from_github('coins')
        
        assert data is None
    
    def test_get_coins_data(self):
        """تست دریافت داده کوین‌ها"""
        with patch.object(self.github_source, '_fetch_from_github') as mock_fetch:
            mock_fetch.return_value = {
                'result': [
                    {'id': 'bitcoin', 'price': 50000},
                    {'id': 'ethereum', 'price': 3000}
                ]
            }
            
            data = self.github_source.get_coins_data()
            
            assert data is not None
            assert len(data['result']) == 2
    
    def test_get_coin_details(self):
        """تست دریافت جزئیات کوین خاص"""
        with patch.object(self.github_source, '_fetch_from_github') as mock_fetch:
            mock_fetch.return_value = {
                'id': 'bitcoin', 
                'price': 50000, 
                'symbol': 'BTC'
            }
            
            data = self.github_source.get_coin_details('bitcoin')
            
            assert data is not None
            assert data['id'] == 'bitcoin'

class TestAPIDataSource:
    """تست منبع داده API"""
    
    def setup_method(self):
        self.api_source = APIDataSource(api_key="test_api_key")
    
    @patch('requests.get')
    def test_make_api_request_success(self, mock_get):
        """تست درخواست موفق به API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result': [{'id': 'bitcoin', 'price': 50000}]
        }
        mock_get.return_value = mock_response
        
        data = self.api_source._make_api_request('coins')
        
        assert data is not None
        assert 'result' in data
    
    @patch('requests.get')
    def test_make_api_request_failure(self, mock_get):
        """تست شکست درخواست به API"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        data = self.api_source._make_api_request('coins')
        
        assert data is None
    
    def test_get_coins_data_with_filters(self):
        """تست دریافت داده کوین‌ها با فیلتر"""
        with patch.object(self.api_source, '_make_api_request') as mock_request:
            mock_request.return_value = {
                'result': [{'id': 'bitcoin', 'price': 50000}],
                'meta': {'page': 1, 'limit': 10}
            }
            
            data = self.api_source.get_coins_data(limit=10, currency='USD')
            
            assert data is not None
            assert 'result' in data
            assert 'meta' in data

class TestSmartDataManager:
    """تست مدیر داده هوشمند"""
    
    def test_initialization_without_api_key(self):
        """تست مقداردهی اولیه بدون کلید API"""
        manager = SmartDataManager()
        
        assert len(manager.sources) >= 1
        assert manager.cache is not None
    
    @patch.object(GitHubDataSource, 'get_coins_data')
    def test_get_coins_data_prioritizes_github(self, mock_github):
        """تست اولویت‌دهی به GitHub در دریافت داده"""
        mock_github.return_value = {
            'result': [{'id': 'bitcoin', 'price': 50000}],
            'meta': {'page': 1}
        }
        
        manager = SmartDataManager()
        data = manager.get_coins_data()
        
        assert data is not None
        mock_github.assert_called_once()
    
    def test_cache_functionality(self):
        """تست عملکرد کش"""
        manager = SmartDataManager()
        
        test_data = {'result': [{'id': 'test', 'price': 100}]}
        manager.cache.set('test_key', test_data)
        cached_data = manager.cache.get('test_key')
        
        assert cached_data == test_data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
