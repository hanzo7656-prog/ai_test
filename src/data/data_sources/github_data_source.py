# 📁 src/data/data_sources/github_data_source.py

import requests
import json
from typing import Dict, List, Any, Optional
from .base_data_source import BaseDataSource
from ...utils.memory_monitor import MemoryMonitor

class GitHubDataSource(BaseDataSource):
    """منبع داده از ریپوی GitHub - اولویت اول"""
    
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data"
        self.memory_monitor = MemoryMonitor()
        self._available_endpoints = self._discover_endpoints()
    
    def _discover_endpoints(self) -> Dict[str, str]:
        """کشف خودکار اندپوینت‌های موجود در ریپو"""
        endpoints = {
            "coins": f"{self.base_url}/coins.json",
            "bitcoin": f"{self.base_url}/bitcoin.json",
            "ethereum": f"{self.base_url}/ethereum.json",
            "charts": f"{self.base_url}/charts",
            "markets": f"{self.base_url}/markets.json",
            "news": f"{self.base_url}/news.json",
            "fear_greed": f"{self.base_url}/fear_greed.json"
        }
        return endpoints
    
    def _fetch_from_github(self, endpoint: str) -> Optional[Any]:
        """دریافت داده از GitHub با مدیریت خطا"""
        try:
            url = self._available_endpoints.get(endpoint)
            if not url:
                return None
                
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.memory_monitor.log_memory_usage(f"github_{endpoint}")
                return data
        except Exception as e:
            print(f"Error fetching from GitHub {endpoint}: {e}")
        return None
    
    def get_coins_data(self, **filters) -> Optional[List[Dict]]:
        """دریافت داده تمام کوین‌ها"""
        return self._fetch_from_github("coins")
    
    def get_coin_details(self, coin_id: str) -> Optional[Dict]:
        """دریافت جزئیات کوین خاص"""
        if coin_id.lower() in ["bitcoin", "btc"]:
            return self._fetch_from_github("bitcoin")
        elif coin_id.lower() in ["ethereum", "eth"]:
            return self._fetch_from_github("ethereum")
        return None
    
    def get_coin_charts(self, coin_id: str, period: str = "all") -> Optional[List[Dict]]:
        """دریافت داده‌های چارت"""
        # ساخت آدرس داینامیک بر اساس coin_id و period
        chart_url = f"{self.base_url}/charts/{coin_id}_{period}.json"
        try:
            response = requests.get(chart_url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def is_available(self) -> bool:
        """بررسی دسترسی به ریپوی GitHub"""
        try:
            test_url = f"{self.base_url}/coins.json"
            response = requests.get(test_url, timeout=5)
            return response.status_code == 200
        except:
            return False
