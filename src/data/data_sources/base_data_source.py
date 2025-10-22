# 📁 src/data/data_sources/base_data_source.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json

class BaseDataSource(ABC):
    """کلاس پایه برای همه منابع داده"""
    
    @abstractmethod
    def get_coins_data(self, **filters) -> Optional[List[Dict]]:
        pass
    
    @abstractmethod
    def get_coin_details(self, coin_id: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_coin_charts(self, coin_id: str, period: str = "all") -> Optional[List[Dict]]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
