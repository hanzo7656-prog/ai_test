# 📁 src/data/processing_pipeline.py

from typing import Dict, List, Any, Optional
from .minimal_processors.feature_selector import FeatureSelector, FeatureConfig
from .minimal_processors.data_compressor import DataCompressor
from .minimal_processors.normalizer import SmartNormalizer, NormalizationConfig
from .minimal_processors.stream_processor import StreamProcessor
from ..utils.performance_tracker import PerformanceTracker

class DataProcessingPipeline:
    """پایپ‌لاین کامل پردازش داده‌های بازار"""
    
    def __init__(self):
        self.feature_selector = FeatureSelector()
        self.data_compressor = DataCompressor(compression_ratio=0.3)
        self.normalizer = SmartNormalizer()
        self.performance_tracker = PerformanceTracker()
        
        self.pipeline_stats = {
            'processed_coins': 0,
            'total_features': 0,
            'memory_savings': 0.0
        }
    
    def process_raw_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """پردازش کامل داده‌های خام"""
        if not raw_data or 'result' not in raw_data:
            return None
        
        print("🔄 Starting data processing pipeline...")
        
        # 1. انتخاب ویژگی‌های حیاتی
        with self.performance_tracker.track("feature_selection"):
            features_data = self.feature_selector.select_features(raw_data)
        
        if not features_data:
            print("❌ Feature selection failed")
            return None
        
        # 2. فشرده‌سازی داده‌ها
        with self.performance_tracker.track("data_compression"):
            compressed_data = self.data_compressor.compress_features(features_data)
        
        # 3. نرمال‌سازی هوشمند
        with self.performance_tracker.track("data_normalization"):
            normalized_data = self.normalizer.fit_transform(compressed_data)
        
        # به‌روزرسانی آمار
        self._update_pipeline_stats(features_data, compressed_data, normalized_data)
        
        print("✅ Data processing completed successfully")
        return normalized_data
    
    def _update_pipeline_stats(self, features_data: Dict, compressed_data: Dict, normalized_data: Dict):
        """به‌روزرسانی آمار پایپ‌لاین"""
        if features_data and 'result' in features_data:
            self.pipeline_stats['processed_coins'] = len(features_data['result'])
            self.pipeline_stats['total_features'] = features_data.get(
                'feature_stats', {}).get('selected_features_per_coin', 0)
        
        if compressed_data and 'compression_stats' in compressed_data:
            self.pipeline_stats['memory_savings'] = compressed_data[
                'compression_stats'].get('savings_percent', 0)
    
    def get_pipeline_info(self) -> Dict:
        """اطلاعات کامل پایپ‌لاین پردازش"""
        feature_summary = self.feature_selector.get_feature_summary()
        performance_summary = self.performance_tracker.get_summary()
        
        return {
            "pipeline_stats": self.pipeline_stats,
            "feature_summary": feature_summary,
            "performance_summary": performance_summary,
            "components": {
                "feature_selector": "✅ Active",
                "data_compressor": "✅ Active", 
                "smart_normalizer": "✅ Active"
            }
        }
