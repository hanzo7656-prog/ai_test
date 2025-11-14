# data_pipeline/data_validator.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """سیستم اعتبارسنجی و کنترل کیفیت داده‌های هوش مصنوعی"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.validation_history = []
        self.quality_metrics = {}
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🔍 Data Validator initialized")

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """مقداردهی اولیه قوانین اعتبارسنجی"""
        return {
            'raw_coins': {
                'required_fields': ['price', 'volume', 'timestamp'],
                'value_ranges': {
                    'price': {'min': 0, 'max': 1000000},
                    'volume': {'min': 0, 'max': 1e12},
                    'market_cap': {'min': 0, 'max': 1e15}
                },
                'data_types': {
                    'price': 'numeric',
                    'volume': 'numeric', 
                    'market_cap': 'numeric',
                    'timestamp': 'datetime'
                },
                'completeness_threshold': 0.8,
                'freshness_threshold': 3600  # 1 hour in seconds
            },
            'raw_exchanges': {
                'required_fields': ['volume', 'pairs_count'],
                'value_ranges': {
                    'volume': {'min': 0, 'max': 1e12},
                    'pairs_count': {'min': 0, 'max': 10000}
                },
                'completeness_threshold': 0.7
            },
            'raw_news': {
                'required_fields': ['title', 'published_at'],
                'value_ranges': {
                    'sentiment': {'min': -1, 'max': 1}
                },
                'completeness_threshold': 0.6,
                'freshness_threshold': 86400  # 24 hours
            },
            'raw_insights': {
                'required_fields': ['analysis', 'confidence'],
                'value_ranges': {
                    'confidence': {'min': 0, 'max': 1},
                    'analysis_depth': {'min': 0, 'max': 1}
                },
                'completeness_threshold': 0.75
            }
        }

    def validate_data_quality(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """اعتبارسنجی کیفیت داده‌های خام"""
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality': 'unknown',
            'source_quality': {},
            'issues_found': 0,
            'validation_summary': {
                'total_sources': 0,
                'valid_sources': 0,
                'invalid_sources': 0
            }
        }
        
        try:
            total_issues = 0
            sources = raw_data.get('sources', {})
            
            validation_report['validation_summary']['total_sources'] = len(sources)
            
            for source_name, source_data in sources.items():
                source_validation = self._validate_single_source(source_name, source_data)
                validation_report['source_quality'][source_name] = source_validation
                
                if source_validation['is_valid']:
                    validation_report['validation_summary']['valid_sources'] += 1
                else:
                    validation_report['validation_summary']['invalid_sources'] += 1
                
                total_issues += len(source_validation.get('issues', []))
            
            validation_report['issues_found'] = total_issues
            
            # تعیین کیفیت کلی
            valid_ratio = validation_report['validation_summary']['valid_sources'] / max(1, validation_report['validation_summary']['total_sources'])
            
            if valid_ratio >= 0.8:
                validation_report['overall_quality'] = 'excellent'
            elif valid_ratio >= 0.6:
                validation_report['overall_quality'] = 'good'
            elif valid_ratio >= 0.4:
                validation_report['overall_quality'] = 'fair'
            else:
                validation_report['overall_quality'] = 'poor'
            
            # ذخیره گزارش اعتبارسنجی
            self.validation_history.append(validation_report)
            self.cache_manager.set_data(
                "utb", 
                "data_validation:latest", 
                validation_report, 
                expire=1800
            )
            
            logger.info(f"✅ Data validation completed: {validation_report['overall_quality']} quality")
            
        except Exception as e:
            logger.error(f"❌ Error in data validation: {e}")
            validation_report['error'] = str(e)
            validation_report['overall_quality'] = 'error'
        
        return validation_report

    def _validate_single_source(self, source_name: str, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """اعتبارسنجی یک منبع داده خاص"""
        validation_result = {
            'source': source_name,
            'is_valid': False,
            'quality_score': 0,
            'issues': [],
            'validation_time': datetime.now().isoformat()
        }
        
        try:
            if source_data.get('status') != 'success':
                validation_result['issues'].append('Source data not available')
                return validation_result
            
            data = source_data.get('data')
            if data is None:
                validation_result['issues'].append('No data received')
                return validation_result
            
            rules = self.validation_rules.get(source_name, {})
            
            # اجرای قوانین اعتبارسنجی
            issues = []
            
            # 1. بررسی کامل بودن داده
            completeness_issues = self._check_completeness(data, rules)
            issues.extend(completeness_issues)
            
            # 2. بررسی محدوده مقادیر
            range_issues = self._check_value_ranges(data, rules)
            issues.extend(range_issues)
            
            # 3. بررسی تازگی داده
            freshness_issues = self._check_freshness(data, rules)
            issues.extend(freshness_issues)
            
            # 4. بررسی سازگاری داده
            consistency_issues = self._check_consistency(data, rules)
            issues.extend(consistency_issues)
            
            validation_result['issues'] = issues
            validation_result['quality_score'] = self._calculate_quality_score(issues, data, rules)
            validation_result['is_valid'] = len(issues) == 0
            
        except Exception as e:
            logger.error(f"❌ Error validating {source_name}: {e}")
            validation_result['issues'].append(f'Validation error: {str(e)}')
        
        return validation_result

    def _check_completeness(self, data: Any, rules: Dict[str, Any]) -> List[str]:
        """بررسی کامل بودن داده"""
        issues = []
        
        try:
            required_fields = rules.get('required_fields', [])
            completeness_threshold = rules.get('completeness_threshold', 0.5)
            
            if isinstance(data, list) and data:
                # بررسی برای لیست داده
                sample_item = data[0] if isinstance(data[0], dict) else {}
                missing_fields = [field for field in required_fields if field not in sample_item]
                
                if missing_fields:
                    issues.append(f'Missing required fields: {missing_fields}')
                
                # بررسی کامل بودن در کل لیست
                total_items = len(data)
                if total_items == 0:
                    issues.append('Empty data list')
                    return issues
                
                complete_items = 0
                for item in data:
                    if isinstance(item, dict):
                        if all(field in item for field in required_fields):
                            complete_items += 1
                
                completeness_ratio = complete_items / total_items
                if completeness_ratio < completeness_threshold:
                    issues.append(f'Low completeness: {completeness_ratio:.1%} (threshold: {completeness_threshold:.0%})')
                    
            elif isinstance(data, dict):
                # بررسی برای دیکشنری تکی
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    issues.append(f'Missing required fields: {missing_fields}')
            else:
                issues.append('Unsupported data format for completeness check')
                
        except Exception as e:
            issues.append(f'Completeness check error: {str(e)}')
        
        return issues

    def _check_value_ranges(self, data: Any, rules: Dict[str, Any]) -> List[str]:
        """بررسی محدوده مقادیر"""
        issues = []
        
        try:
            value_ranges = rules.get('value_ranges', {})
            
            if not value_ranges:
                return issues
            
            def check_item_values(item):
                item_issues = []
                for field, range_config in value_ranges.items():
                    if field in item:
                        value = item[field]
                        if isinstance(value, (int, float)):
                            min_val = range_config.get('min')
                            max_val = range_config.get('max')
                            
                            if min_val is not None and value < min_val:
                                item_issues.append(f'{field} below minimum: {value} < {min_val}')
                            if max_val is not None and value > max_val:
                                item_issues.append(f'{field} above maximum: {value} > {max_val}')
                return item_issues
            
            if isinstance(data, list):
                for i, item in enumerate(data[:10]):  # بررسی 10 نمونه اول
                    if isinstance(item, dict):
                        item_issues = check_item_values(item)
                        for issue in item_issues:
                            issues.append(f'Item {i}: {issue}')
            elif isinstance(data, dict):
                issues.extend(check_item_values(data))
                
        except Exception as e:
            issues.append(f'Value range check error: {str(e)}')
        
        return issues

    def _check_freshness(self, data: Any, rules: Dict[str, Any]) -> List[str]:
        """بررسی تازگی داده"""
        issues = []
        
        try:
            freshness_threshold = rules.get('freshness_threshold')
            if not freshness_threshold:
                return issues
            
            timestamp_fields = ['timestamp', 'published_at', 'created_at', 'time']
            
            def extract_timestamp(item):
                for field in timestamp_fields:
                    if field in item:
                        return item[field]
                return None
            
            if isinstance(data, list) and data:
                latest_timestamp = None
                
                for item in data[:5]:  # بررسی 5 نمونه اول
                    if isinstance(item, dict):
                        ts_str = extract_timestamp(item)
                        if ts_str:
                            try:
                                # تبدیل رشته به datetime
                                if 'Z' in ts_str:
                                    ts_str = ts_str.replace('Z', '+00:00')
                                item_time = datetime.fromisoformat(ts_str)
                                
                                if latest_timestamp is None or item_time > latest_timestamp:
                                    latest_timestamp = item_time
                            except:
                                pass
                
                if latest_timestamp:
                    time_diff = (datetime.now() - latest_timestamp).total_seconds()
                    if time_diff > freshness_threshold:
                        issues.append(f'Data is stale: {time_diff/3600:.1f}h old (threshold: {freshness_threshold/3600:.1f}h)')
                else:
                    issues.append('No valid timestamps found')
                    
        except Exception as e:
            issues.append(f'Freshness check error: {str(e)}')
        
        return issues

    def _check_consistency(self, data: Any, rules: Dict[str, Any]) -> List[str]:
        """بررسی سازگاری داده"""
        issues = []
        
        try:
            if isinstance(data, list) and len(data) > 1:
                # بررسی سازگاری بین آیتم‌ها
                first_item = data[0]
                
                for i, item in enumerate(data[1:5]):  # بررسی 4 نمونه بعدی
                    if isinstance(item, dict) and isinstance(first_item, dict):
                        # بررسی کلیدها
                        if set(item.keys()) != set(first_item.keys()):
                            issues.append(f'Inconsistent keys at item {i+1}')
                        
                        # بررسی انواع داده
                        for key in first_item.keys():
                            if key in item:
                                if type(first_item[key]) != type(item[key]):
                                    issues.append(f'Inconsistent type for {key} at item {i+1}')
            
        except Exception as e:
            issues.append(f'Consistency check error: {str(e)}')
        
        return issues

    def _calculate_quality_score(self, issues: List[str], data: Any, rules: Dict[str, Any]) -> float:
        """محاسبه نمره کیفیت داده"""
        try:
            base_score = 100.0
            
            # کسر بر اساس تعداد issues
            issue_penalty = len(issues) * 10
            base_score -= issue_penalty
            
            # پاداش برای حجم داده
            if isinstance(data, list):
                data_size = len(data)
                if data_size > 100:
                    base_score += 10
                elif data_size > 50:
                    base_score += 5
                elif data_size == 0:
                    base_score = 0
            
            return max(0, min(100, base_score))
            
        except:
            return 0.0

    def get_validation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """دریافت تاریخچه اعتبارسنجی"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            report for report in self.validation_history
            if datetime.fromisoformat(report['timestamp']) > cutoff_time
        ]

    def get_data_quality_trends(self) -> Dict[str, Any]:
        """دریافت روندهای کیفیت داده"""
        try:
            recent_reports = self.get_validation_history(24)
            
            if not recent_reports:
                return {'message': 'No validation data available'}
            
            trends = {
                'analysis_period': '24h',
                'timestamp': datetime.now().isoformat(),
                'quality_trends': {},
                'source_performance': {},
                'common_issues': {}
            }
            
            # تحلیل روند کیفیت
            quality_scores = []
            for report in recent_reports:
                quality_scores.append({
                    'time': report['timestamp'],
                    'score': self._quality_score_to_numeric(report['overall_quality'])
                })
            
            trends['quality_trends'] = {
                'average_score': np.mean([q['score'] for q in quality_scores]) if quality_scores else 0,
                'trend_direction': 'improving' if len(quality_scores) > 1 and quality_scores[-1]['score'] > quality_scores[0]['score'] else 'declining',
                'stability': np.std([q['score'] for q in quality_scores]) if quality_scores else 0
            }
            
            # تحلیل عملکرد منابع
            source_stats = {}
            for report in recent_reports:
                for source_name, source_quality in report.get('source_quality', {}).items():
                    if source_name not in source_stats:
                        source_stats[source_name] = []
                    source_stats[source_name].append(source_quality.get('quality_score', 0))
            
            for source_name, scores in source_stats.items():
                trends['source_performance'][source_name] = {
                    'average_score': np.mean(scores),
                    'reliability': 'high' if np.mean(scores) > 80 else 'medium' if np.mean(scores) > 60 else 'low'
                }
            
            # تحلیل issues رایج
            all_issues = []
            for report in recent_reports:
                for source_quality in report.get('source_quality', {}).values():
                    all_issues.extend(source_quality.get('issues', []))
            
            from collections import Counter
            common_issues = Counter(all_issues).most_common(5)
            trends['common_issues'] = dict(common_issues)
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ Error analyzing quality trends: {e}")
            return {'error': str(e)}

    def _quality_score_to_numeric(self, quality: str) -> float:
        """تبدیل کیفیت کیفی به عددی"""
        mapping = {
            'excellent': 90,
            'good': 75,
            'fair': 60,
            'poor': 40,
            'unknown': 50,
            'error': 0
        }
        return mapping.get(quality, 50)

# نمونه global
data_validator = DataValidator()
