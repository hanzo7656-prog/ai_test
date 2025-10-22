# 📁 src/data/data_manager.py

from typing import Dict, List, Any, Optional
from .data_sources.base_data_source import BaseDataSource
from .data_sources.github_data_source import GitHubDataSource
from .data_sources.api_data_source import APIDataSource
from .cache.lru_cache import LRUCache
from ..utils.performance_tracker import PerformanceTracker

class SmartDataManager:
    """مدیریت هوشمند منابع داده با اولویت‌بندی"""
    
    def __init__(self, api_key: str = None):
        self.sources: List[BaseDataSource] = []
        self.cache = LRUCache(maxsize=200)
        self.performance_tracker = PerformanceTracker()
        
        # اولویت‌بندی منابع داده
        self._initialize_sources(api_key)
    
    def _initialize_sources(self, api_key: str):
        """مقداردهی منابع داده به ترتیب اولویت"""
        # 1. اولویت با GitHub
        github_source = GitHubDataSource()
        if github_source.is_available():
            self.sources.append(github_source)
            print("✅ GitHub data source initialized")
        
        # 2. سپس API
        if api_key:
            api_source = APIDataSource(api_key)
            if api_source.is_available():
                self.sources.append(api_source)
                print("✅ API data source initialized")
        
        if not self.sources:
            print("⚠️ No data sources available!")
    
    def get_coins_data(self, use_cache: bool = True, **filters) -> Optional[List[Dict]]:
        """دریافت داده کوین‌ها از بهترین منبع موجود"""
        cache_key = f"coins_{str(filters)}"
        
        # بررسی کش
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        # جستجو در منابع به ترتیب اولویت
        for source in self.sources:
            with self.performance_tracker.track(f"get_coins_{source.__class__.__name__}"):
                data = source.get_coins_data(**filters)
                
            if data and 'result' in data:
                # ذخیره در کش
                self.cache.set(cache_key, data)
                return data
        
        return None
    
    def get_coin_details(self, coin_id: str, use_cache: bool = True) -> Optional[Dict]:
        """دریافت جزئیات کوین خاص"""
        cache_key = f"coin_{coin_id}"
        
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        for source in self.sources:
            with self.performance_tracker.track(f"get_coin_{source.__class__.__name__}"):
                data = source.get_coin_details(coin_id)
                
            if data:
                self.cache.set(cache_key, data)
                return data
        
        return None
    
    def get_coin_charts(self, coin_id: str, period: str = "all", use_cache: bool = True) -> Optional[List[Dict]]:
        """دریافت داده‌های چارت"""
        cache_key = f"charts_{coin_id}_{period}"
        
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        for source in self.sources:
            with self.performance_tracker.track(f"get_charts_{source.__class__.__name__}"):
                data = source.get_coin_charts(coin_id, period)
                
            if data:
                self.cache.set(cache_key, data)
                return data
        
        return None
    
    def get_source_stats(self) -> Dict:
        """آمار منابع داده"""
        return {
            "total_sources": len(self.sources),
            "sources_available": [source.__class__.__name__ for source in self.sources],
            "cache_size": self.cache.current_size,
            "performance_metrics": self.performance_tracker.get_summary()
        }
