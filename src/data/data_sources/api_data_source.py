# 📁 src/data/data_sources/api_data_source.py

import requests
import time
from typing import Dict, List, Any, Optional
from .base_data_source import BaseDataSource
from ...utils.memory_monitor import MemoryMonitor

class APIDataSource(BaseDataSource):
    """منبع داده از CoinStats API - اولویت دوم"""
    
    def __init__(self, api_key: str):
        self.base_url = "https://openapiv1.coinstats.app"
        self.api_key = api_key
        self.memory_monitor = MemoryMonitor()
        self.last_request_time = 0
        self.min_interval = 0.1  # 100ms بین درخواست‌ها
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """ارسال درخواست به API با مدیریت نرخ و خطا"""
        # رعایت فاصله زمانی بین درخواست‌ها
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"X-API-KEY": self.api_key}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.memory_monitor.log_memory_usage(f"api_{endpoint}")
                return data
            else:
                print(f"API Error {endpoint}: {response.status_code}")
                
        except Exception as e:
            print(f"Request failed {endpoint}: {e}")
            
        return None
    
    def get_coins_data(self, **filters) -> Optional[List[Dict]]:
        """دریافت داده تمام کوین‌ها از API"""
        params = {}
        
        # تبدیل فیلترها به پارامترهای API
        if 'limit' in filters:
            params['limit'] = filters['limit']
        if 'currency' in filters:
            params['currency'] = filters['currency']
        if 'sort_by' in filters:
            params['sortBy'] = filters['sort_by']
            
        return self._make_api_request("coins", params)
    
    def get_coin_details(self, coin_id: str) -> Optional[Dict]:
        """دریافت جزئیات کوین خاص از API"""
        return self._make_api_request(f"coins/{coin_id}")
    
    def get_coin_charts(self, coin_id: str, period: str = "all") -> Optional[List[Dict]]:
        """دریافت داده‌های چارت از API"""
        params = {"period": period}
        return self._make_api_request(f"coins/{coin_id}/charts", params)
    
    def is_available(self) -> bool:
        """بررسی دسترسی به API"""
        try:
            test_data = self._make_api_request("coins", {"limit": 1})
            return test_data is not None and 'result' in test_data
        except:
            return False
