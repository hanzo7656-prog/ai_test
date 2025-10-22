# 📁 src/data/minimal_processors/feature_selector.py

import numpy as np
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from ...utils.memory_monitor import MemoryMonitor

@dataclass
class FeatureConfig:
    """پیکربندی ویژگی‌های حیاتی برای تحلیل بازار"""
    # ویژگی‌های قیمتی (6 مورد)
    price_features: Set[str] = None
    # ویژگی‌های حجم و نقدینگی (4 مورد)
    volume_features: Set[str] = None
    # ویژگی‌های تغییرات قیمتی (5 مورد)
    change_features: Set[str] = None
    # ویژگی‌های عرضه و بازار (3 مورد)
    supply_features: Set[str] = None
    # ویژگی‌های تکنیکال (2 مورد)
    technical_features: Set[str] = None
    
    def __post_init__(self):
        if self.price_features is None:
            self.price_features = {
                'price', 'priceBtc', 'marketCap', 
                'fullyDilutedValuation', 'high', 'low'
            }
        if self.volume_features is None:
            self.volume_features = {
                'volume', 'liquidityScore', 'volatilityScore', 'marketCapScore'
            }
        if self.change_features is None:
            self.change_features = {
                'priceChange1h', 'priceChange1d', 'priceChange1w',
                'avgChange', 'priceChange24h'
            }
        if self.supply_features is None:
            self.supply_features = {
                'availableSupply', 'totalSupply', 'circulatingSupply'
            }
        if self.technical_features is None:
            self.technical_features = {
                'rank', 'riskScore'
            }
    
    @property
    def all_features(self) -> Set[str]:
        """تمامی ۲۰ ویژگی حیاتی"""
        return (self.price_features | self.volume_features | 
                self.change_features | self.supply_features | 
                self.technical_features)

class FeatureSelector:
    """انتخابگر ۲۰ ویژگی حیاتی از داده‌های خام"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.memory_monitor = MemoryMonitor()
        self.selected_features_count = 0
        
    def select_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """انتخاب ویژگی‌های حیاتی از داده خام"""
        if not raw_data or 'result' not in raw_data:
            return {}
            
        processed_coins = []
        
        for coin in raw_data['result']:
            if not isinstance(coin, dict):
                continue
                
            selected_features = self._extract_critical_features(coin)
            if selected_features:
                processed_coins.append(selected_features)
        
        result = {
            'meta': raw_data.get('meta', {}),
            'result': processed_coins,
            'feature_stats': {
                'total_coins': len(processed_coins),
                'selected_features_per_coin': self.selected_features_count,
                'memory_usage': self.memory_monitor.get_current_usage()
            }
        }
        
        self.memory_monitor.log_memory_usage("feature_selection")
        return result
    
    def _extract_critical_features(self, coin_data: Dict) -> Dict:
        """استخراج ۲۰ ویژگی حیاتی از هر کوین"""
        critical_features = {}
        
        # استخراج ویژگی‌های قیمتی
        for feature in self.config.price_features:
            if feature in coin_data:
                critical_features[f"price_{feature}"] = self._safe_float(coin_data[feature])
        
        # استخراج ویژگی‌های حجم
        for feature in self.config.volume_features:
            if feature in coin_data:
                critical_features[f"volume_{feature}"] = self._safe_float(coin_data[feature])
        
        # استخراج ویژگی‌های تغییرات
        for feature in self.config.change_features:
            if feature in coin_data:
                critical_features[f"change_{feature}"] = self._safe_float(coin_data[feature])
        
        # استخراج ویژگی‌های عرضه
        for feature in self.config.supply_features:
            if feature in coin_data:
                critical_features[f"supply_{feature}"] = self._safe_float(coin_data[feature])
        
        # استخراج ویژگی‌های تکنیکال
        for feature in self.config.technical_features:
            if feature in coin_data:
                critical_features[f"tech_{feature}"] = self._safe_float(coin_data[feature])
        
        # افزودن شناسه کوین
        critical_features['coin_id'] = coin_data.get('id', '')
        critical_features['symbol'] = coin_data.get('symbol', '')
        
        self.selected_features_count = len(critical_features) - 2  # بدون احتساب شناسه و سمبل
        
        return critical_features
    
    def _safe_float(self, value: Any) -> float:
        """تبدیل ایمن به float"""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_feature_summary(self) -> Dict:
        """خلاصه‌ای از ویژگی‌های انتخاب‌شده"""
        return {
            "total_critical_features": len(self.config.all_features),
            "feature_categories": {
                "price": len(self.config.price_features),
                "volume": len(self.config.volume_features),
                "change": len(self.config.change_features),
                "supply": len(self.config.supply_features),
                "technical": len(self.config.technical_features)
            },
            "memory_usage": self.memory_monitor.get_usage_stats()
        }
