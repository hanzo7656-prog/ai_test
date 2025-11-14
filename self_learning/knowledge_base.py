# self_learning/knowledge_base.py
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """پایگاه دانش و حافظه برای سیستم هوش مصنوعی"""
    
    def __init__(self):
        self.model_knowledge = defaultdict(dict)
        self.training_patterns = defaultdict(list)
        self.performance_insights = defaultdict(dict)
        self.experience_buffer = deque(maxlen=5000)
        
        # دانش domain-specific
        self.market_patterns = {}
        self.trading_rules = {}
        self.risk_profiles = {}
        
        # ارتباط با سیستم کش
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        # بارگذاری دانش موجود
        self._load_existing_knowledge()
        
        logger.info("📚 Knowledge Base initialized")

    def _load_existing_knowledge(self):
        """بارگذاری دانش موجود از کش"""
        try:
            # بارگذاری دانش مدل‌ها
            model_knowledge = self.cache_manager.get_data("utb", "model_knowledge")
            if model_knowledge:
                self.model_knowledge.update(model_knowledge)
            
            # بارگذاری الگوهای آموزشی
            training_patterns = self.cache_manager.get_data("utb", "training_patterns") 
            if training_patterns:
                self.training_patterns.update(training_patterns)
                
            logger.info("✅ Existing knowledge loaded from cache")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing knowledge: {e}")

    def save_model_experience(self, model_name: str, experience: Dict[str, Any]):
        """ذخیره تجربه آموزشی مدل"""
        try:
            experience_id = hashlib.md5(
                f"{model_name}_{datetime.now().timestamp()}".encode()
            ).hexdigest()
            
            experience_data = {
                'experience_id': experience_id,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'data': experience,
                'type': experience.get('type', 'training')
            }
            
            # ذخیره در بافر تجربیات
            self.experience_buffer.append(experience_data)
            
            # ذخیره در دانش مدل
            if model_name not in self.model_knowledge:
                self.model_knowledge[model_name] = {
                    'total_experiences': 0,
                    'last_updated': datetime.now().isoformat(),
                    'experiences': []
                }
            
            self.model_knowledge[model_name]['experiences'].append(experience_data)
            self.model_knowledge[model_name]['total_experiences'] += 1
            self.model_knowledge[model_name]['last_updated'] = datetime.now().isoformat()
            
            # آنالیز الگوها
            self._analyze_training_patterns(model_name, experience)
            
            # ذخیره در کش
            self._save_knowledge_to_cache()
            
            logger.debug(f"💾 Saved experience for {model_name}: {experience_id}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model experience: {e}")

    def _analyze_training_patterns(self, model_name: str, experience: Dict[str, Any]):
        """آنالیز الگوهای آموزشی از تجربیات"""
        try:
            pattern_key = f"{model_name}_{experience.get('type', 'general')}"
            
            pattern_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': experience.get('performance_metrics', {}),
                'training_config': experience.get('training_config', {}),
                'data_characteristics': experience.get('data_characteristics', {})
            }
            
            self.training_patterns[pattern_key].append(pattern_data)
            
            # حفظ فقط ۱۰۰ نمونه اخیر برای هر الگو
            if len(self.training_patterns[pattern_key]) > 100:
                self.training_patterns[pattern_key] = self.training_patterns[pattern_key][-100:]
                
            # استخراج insight‌ها
            self._extract_performance_insights(model_name, pattern_data)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing training patterns: {e}")

    def _extract_performance_insights(self, model_name: str, pattern_data: Dict[str, Any]):
        """استخراج insight‌های عملکرد از الگوها"""
        try:
            insights = self.performance_insights.get(model_name, {})
            
            metrics = pattern_data.get('performance_metrics', {})
            config = pattern_data.get('training_config', {})
            
            # محاسبه بهبود عملکرد
            if 'accuracy' in metrics and 'previous_accuracy' in metrics:
                improvement = metrics['accuracy'] - metrics['previous_accuracy']
                
                if improvement > insights.get('best_improvement', 0):
                    insights['best_improvement'] = improvement
                    insights['best_config'] = config
                    insights['best_timestamp'] = pattern_data['timestamp']
            
            # ردیابی بهترین تنظیمات
            if metrics.get('accuracy', 0) > insights.get('best_accuracy', 0):
                insights['best_accuracy'] = metrics['accuracy']
                insights['best_accuracy_config'] = config
            
            self.performance_insights[model_name] = insights
            
        except Exception as e:
            logger.error(f"❌ Error extracting performance insights: {e}")

    def get_model_knowledge(self, model_name: str) -> Dict[str, Any]:
        """دریافت دانش ذخیره شده برای یک مدل"""
        return self.model_knowledge.get(model_name, {
            'total_experiences': 0,
            'last_updated': None,
            'experiences': []
        })

    def find_similar_experiences(self, model_name: str, current_context: Dict[str, Any], 
                               max_results: int = 5) -> List[Dict[str, Any]]:
        """پیداکردن تجربیات مشابه برای یادگیری انتقالی"""
        try:
            similar_experiences = []
            model_experiences = self.model_knowledge.get(model_name, {}).get('experiences', [])
            
            for experience in model_experiences[-100:]:  # فقط ۱۰۰ تجربه اخیر
                similarity_score = self._calculate_context_similarity(
                    current_context, 
                    experience['data']
                )
                
                if similarity_score > 0.7:  threshold 
                    similar_experiences.append({
                        'experience': experience,
                        'similarity_score': similarity_score,
                        'relevance': self._calculate_relevance(experience, current_context)
                    })
            
            # مرتب‌سازی بر اساس similarity و relevance
            similar_experiences.sort(key=lambda x: x['similarity_score'] * x['relevance'], reverse=True)
            
            return similar_experiences[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Error finding similar experiences: {e}")
            return []

    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """محاسبه شباهت بین دو context"""
        try:
            similarity = 0.0
            compared_features = 0
            
            # مقایسه ویژگی‌های عددی
            numeric_features = ['data_size', 'feature_count', 'training_time', 'accuracy']
            
            for feature in numeric_features:
                if feature in context1 and feature in context2:
                    val1 = context1[feature] if context1[feature] else 0
                    val2 = context2[feature] if context2[feature] else 0
                    
                    if val1 + val2 > 0:
                        similarity += 1 - abs(val1 - val2) / max(val1, val2)
                        compared_features += 1
            
            # مقایسه ویژگی‌های کیفی
            qualitative_features = ['data_type', 'market_condition', 'training_strategy']
            
            for feature in qualitative_features:
                if feature in context1 and feature in context2:
                    if context1[feature] == context2[feature]:
                        similarity += 1.0
                    compared_features += 1
            
            return similarity / compared_features if compared_features > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating context similarity: {e}")
            return 0.0

    def _calculate_relevance(self, experience: Dict[str, Any], current_context: Dict[str, Any]) -> float:
        """محاسبه relevance تجربه"""
        try:
            # relevance بر اساس تازگی و performance
            experience_time = datetime.fromisoformat(experience['timestamp'])
            time_diff = (datetime.now() - experience_time).total_seconds()
            
            # تجربیات جدیدتر relevance بیشتری دارند
            time_relevance = max(0, 1 - (time_diff / (30 * 24 * 3600)))  # 30 روز
            
            # تجربیات با performance بهتر relevance بیشتری دارند
            performance = experience['data'].get('performance_metrics', {}).get('accuracy', 0.5)
            performance_relevance = performance
            
            return (time_relevance + performance_relevance) / 2
            
        except Exception as e:
            logger.error(f"❌ Error calculating relevance: {e}")
            return 0.5

    def get_training_recommendations(self, model_name: str, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """دریافت توصیه‌های آموزشی بر اساس دانش موجود"""
        try:
            recommendations = []
            
            # تحلیل الگوهای موفق گذشته
            successful_patterns = self._find_successful_patterns(model_name, current_performance)
            
            for pattern in successful_patterns:
                recommendation = {
                    'type': 'training_strategy',
                    'confidence': pattern['success_score'],
                    'suggested_config': pattern['config'],
                    'expected_improvement': pattern['improvement'],
                    'reasoning': pattern['reasoning'],
                    'based_on_experiences': pattern['experience_count']
                }
                recommendations.append(recommendation)
            
            # توصیه‌های based on performance gaps
            performance_recommendations = self._generate_performance_recommendations(
                model_name, current_performance
            )
            recommendations.extend(performance_recommendations)
            
            # مرتب‌سازی بر اساس confidence
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            return recommendations[:10]  # حداکثر ۱۰ توصیه
            
        except Exception as e:
            logger.error(f"❌ Error generating training recommendations: {e}")
            return []

    def _find_successful_patterns(self, model_name: str, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """پیداکردن الگوهای آموزشی موفق"""
        successful_patterns = []
        
        pattern_key = f"{model_name}_training"
        patterns = self.training_patterns.get(pattern_key, [])
        
        for pattern in patterns[-50:]:  # ۵۰ الگوی اخیر
            metrics = pattern.get('performance_metrics', {})
            
            # محاسبه امتیاز موفقیت
            success_score = self._calculate_success_score(metrics, current_performance)
            
            if success_score > 0.7:
                successful_patterns.append({
                    'config': pattern.get('training_config', {}),
                    'success_score': success_score,
                    'improvement': metrics.get('accuracy', 0) - current_performance.get('accuracy', 0),
                    'reasoning': f"این تنظیمات قبلاً منجر به دقت {metrics.get('accuracy', 0):.3f} شده است",
                    'experience_count': 1
                })
        
        return successful_patterns

    def _calculate_success_score(self, historical_metrics: Dict[str, Any], 
                               current_metrics: Dict[str, Any]) -> float:
        """محاسبه امتیاز موفقیت الگو"""
        score = 0.0
        factors = 0
        
        # مقایسه دقت
        if 'accuracy' in historical_metrics and 'accuracy' in current_metrics:
            accuracy_improvement = historical_metrics['accuracy'] - current_metrics['accuracy']
            if accuracy_improvement > 0:
                score += min(1.0, accuracy_improvement * 2)  # بهبود ۵٪ = امتیاز ۱
            factors += 1
        
        # سایر فاکتورها
        if 'training_time' in historical_metrics:
            # الگوهای سریع‌تر امتیاز بیشتری می‌گیرند
            time_score = 1.0 / (1.0 + historical_metrics['training_time'] / 3600)  # نرمال‌سازی بر اساس ساعت
            score += time_score
            factors += 1
        
        return score / factors if factors > 0 else 0.0

    def _generate_performance_recommendations(self, model_name: str, 
                                           current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تولید توصیه‌های based on شکاف عملکرد"""
        recommendations = []
        
        insights = self.performance_insights.get(model_name, {})
        
        # اگر بهترین عملکرد خیلی بهتر از عملکرد فعلی است
        best_accuracy = insights.get('best_accuracy', 0)
        current_accuracy = current_performance.get('accuracy', 0)
        
        if best_accuracy - current_accuracy > 0.1:  # شکاف ۱۰٪
            recommendations.append({
                'type': 'performance_gap',
                'confidence': 0.8,
                'suggested_action': 'use_best_known_config',
                'config': insights.get('best_accuracy_config', {}),
                'expected_improvement': best_accuracy - current_accuracy,
                'reasoning': f'شکاف عملکرد {best_accuracy - current_accuracy:.3f} - استفاده از تنظیمات بهینه گذشته'
            })
        
        return recommendations

    def save_market_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]):
        """ذخیره الگوی بازار جدید"""
        try:
            pattern_id = hashlib.md5(pattern_name.encode()).hexdigest()
            
            market_pattern = {
                'pattern_id': pattern_id,
                'name': pattern_name,
                'data': pattern_data,
                'discovered_at': datetime.now().isoformat(),
                'confidence': pattern_data.get('confidence', 0.5),
                'occurrence_count': 1
            }
            
            # اگر الگو از قبل وجود دارد، occurrence را افزایش بده
            if pattern_id in self.market_patterns:
                existing = self.market_patterns[pattern_id]
                existing['occurrence_count'] += 1
                existing['confidence'] = max(existing['confidence'], pattern_data.get('confidence', 0.5))
                existing['last_seen'] = datetime.now().isoformat()
            else:
                self.market_patterns[pattern_id] = market_pattern
            
            # ذخیره در کش
            self.cache_manager.set_data("utb", f"market_pattern:{pattern_id}", market_pattern, expire=86400)
            
            logger.info(f"🔍 Saved market pattern: {pattern_name}")
            
        except Exception as e:
            logger.error(f"❌ Error saving market pattern: {e}")

    def find_relevant_market_patterns(self, current_market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """پیداکردن الگوهای بازار مرتبط"""
        relevant_patterns = []
        
        for pattern_id, pattern in self.market_patterns.items():
            relevance = self._calculate_market_relevance(pattern, current_market_data)
            
            if relevance > 0.6:  # threshold
                relevant_patterns.append({
                    'pattern': pattern,
                    'relevance': relevance,
                    'predicted_outcome': pattern['data'].get('expected_outcome', 'unknown')
                })
        
        # مرتب‌سازی بر اساس relevance
        relevant_patterns.sort(key=lambda x: x['relevance'], reverse=True)
        
        return relevant_patterns[:5]  # ۵ الگوی برتر

    def _calculate_market_relevance(self, pattern: Dict[str, Any], 
                                  current_data: Dict[str, Any]) -> float:
        """محاسبه relevance الگوی بازار"""
        # این یک پیاده‌سازی ساده است
        # در عمل باید از similarity measures پیشرفته استفاده شود
        return pattern.get('confidence', 0.5)  # placeholder

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه دانش موجود"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_models_with_knowledge': len(self.model_knowledge),
            'total_experiences': sum(
                model_data.get('total_experiences', 0) 
                for model_data in self.model_knowledge.values()
            ),
            'total_training_patterns': sum(
                len(patterns) for patterns in self.training_patterns.values()
            ),
            'total_market_patterns': len(self.market_patterns),
            'knowledge_quality_metrics': {
                'avg_experiences_per_model': self._calculate_avg_experiences(),
                'knowledge_freshness': self._calculate_knowledge_freshness(),
                'pattern_effectiveness': self._calculate_pattern_effectiveness()
            }
        }

    def _calculate_avg_experiences(self) -> float:
        """محاسبه میانگین تجربیات per model"""
        if not self.model_knowledge:
            return 0.0
        total = sum(model_data.get('total_experiences', 0) for model_data in self.model_knowledge.values())
        return total / len(self.model_knowledge)

    def _calculate_knowledge_freshness(self) -> float:
        """محاسبه تازگی دانش"""
        if not self.model_knowledge:
            return 0.0
        
        now = datetime.now()
        freshness_scores = []
        
        for model_data in self.model_knowledge.values():
            last_updated = datetime.fromisoformat(model_data.get('last_updated', now.isoformat()))
            days_old = (now - last_updated).total_seconds() / (24 * 3600)
            freshness = max(0, 1 - (days_old / 30))  # دانش قدیمی‌تر از ۳۰ روز امتیاز کمتری
            freshness_scores.append(freshness)
        
        return np.mean(freshness_scores) if freshness_scores else 0.0

    def _calculate_pattern_effectiveness(self) -> float:
        """محاسبه اثربخشی الگوها"""
        # این یک پیاده‌سازی ساده است
        # در عمل باید بر اساس نتاقع الگوها محاسبه شود
        return 0.75  # placeholder

    def _save_knowledge_to_cache(self):
        """ذخیره دانش در کش"""
        try:
            knowledge_data = {
                'model_knowledge': dict(self.model_knowledge),
                'training_patterns': dict(self.training_patterns),
                'performance_insights': dict(self.performance_insights),
                'market_patterns': self.market_patterns,
                'last_saved': datetime.now().isoformat()
            }
            
            self.cache_manager.set_data("utb", "knowledge_base", knowledge_data, expire=7200)  # 2 hours
            
        except Exception as e:
            logger.error(f"❌ Error saving knowledge to cache: {e}")

    def export_knowledge(self, export_path: str = None) -> Dict[str, Any]:
        """خروجی گرفتن از دانش برای backup یا transfer"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'knowledge_base': {
                'model_knowledge': dict(self.model_knowledge),
                'training_patterns': dict(self.training_patterns),
                'performance_insights': dict(self.performance_insights),
                'market_patterns': self.market_patterns
            },
            'summary': self.get_knowledge_summary()
        }
        
        if export_path:
            try:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Knowledge exported to {export_path}")
            except Exception as e:
                logger.error(f"❌ Error exporting knowledge to file: {e}")
        
        return export_data

# نمونه global
knowledge_base = None

def initialize_knowledge_base():
    """مقداردهی اولیه knowledge base"""
    global knowledge_base
    knowledge_base = KnowledgeBase()
    return knowledge_base
