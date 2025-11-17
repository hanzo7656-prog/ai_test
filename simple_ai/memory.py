import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class AIMemoryManager:
    """مدیریت حافظه و ذخیره‌سازی دانش هوش مصنوعی"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.memory_stats = {
            'total_knowledge_items': 0,
            'memory_usage_mb': 0,
            'last_saved': None,
            'access_count': 0
        }
    
    def save_knowledge(self, key: str, knowledge: Dict, category: str = "general"):
        """ذخیره دانش در حافظه"""
        try:
            knowledge_item = {
                'data': knowledge,
                'metadata': {
                    'category': category,
                    'created_at': datetime.now().isoformat(),
                    'access_count': 0,
                    'importance_score': self._calculate_importance(knowledge)
                }
            }
            
            self.knowledge_base[key] = knowledge_item
            self.memory_stats['total_knowledge_items'] = len(self.knowledge_base)
            self.memory_stats['memory_usage_mb'] = self._calculate_memory_usage()
            self.memory_stats['last_saved'] = datetime.now().isoformat()
            
            print(f"💾 Knowledge saved: {key} (Category: {category})")
            
        except Exception as e:
            print(f"❌ Error saving knowledge: {e}")
    
    def load_knowledge(self, key: str) -> Optional[Dict]:
        """بارگذاری دانش از حافظه"""
        try:
            if key in self.knowledge_base:
                knowledge_item = self.knowledge_base[key]
                knowledge_item['metadata']['access_count'] += 1
                knowledge_item['metadata']['last_accessed'] = datetime.now().isoformat()
                
                self.memory_stats['access_count'] += 1
                
                print(f"🔍 Knowledge loaded: {key}")
                return knowledge_item['data']
            else:
                print(f"⚠️ Knowledge not found: {key}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading knowledge: {e}")
            return None
    
    def _calculate_importance(self, knowledge: Dict) -> float:
        """محاسبه امتیاز اهمیت دانش"""
        score = 0.0
        
        # معیارهای اهمیت
        if isinstance(knowledge, dict):
            # دانش ساختاریافته امتیاز بیشتری دارد
            score += 0.3
        
        if len(str(knowledge)) > 100:
            # دانش حجیم‌تر امتیاز بیشتری دارد
            score += 0.2
        
        # امتیاز مبتنی بر نوع داده
        data_type = type(knowledge).__name__
        if data_type in ['list', 'dict']:
            score += 0.2
        elif data_type == 'str':
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_memory_usage(self) -> float:
        """محاسبه استفاده از حافظه"""
        try:
            memory_size = len(json.dumps(self.knowledge_base, ensure_ascii=False).encode('utf-8'))
            return round(memory_size / (1024 * 1024), 2)  # تبدیل به مگابایت
        except:
            return 0.0
    
    def search_knowledge(self, query: str, category: str = None) -> List[Dict]:
        """جستجو در پایگاه دانش"""
        results = []
        
        for key, item in self.knowledge_base.items():
            # جستجو در کلیدها
            if query.lower() in key.lower():
                results.append({
                    'key': key,
                    'data': item['data'],
                    'metadata': item['metadata'],
                    'match_type': 'key_match'
                })
                continue
            
            # جستجو در داده‌ها (برای رشته‌ها)
            if isinstance(item['data'], str) and query.lower() in item['data'].lower():
                results.append({
                    'key': key,
                    'data': item['data'],
                    'metadata': item['metadata'],
                    'match_type': 'content_match'
                })
        
        # فیلتر بر اساس دسته اگر مشخص شده باشد
        if category:
            results = [r for r in results if r['metadata']['category'] == category]
        
        # مرتب‌سازی بر اساس اهمیت
        results.sort(key=lambda x: x['metadata']['importance_score'], reverse=True)
        
        return results
    
    def get_knowledge_by_category(self, category: str) -> List[Dict]:
        """دریافت تمام دانش یک دسته"""
        category_items = []
        
        for key, item in self.knowledge_base.items():
            if item['metadata']['category'] == category:
                category_items.append({
                    'key': key,
                    'data': item['data'],
                    'metadata': item['metadata']
                })
        
        return category_items
    
    def cleanup_memory(self, max_items: int = 1000):
        """پاک‌سازی حافظه در صورت لزوم"""
        if len(self.knowledge_base) <= max_items:
            return
        
        # مرتب‌سازی بر اساس امتیاز اهمیت و تعداد دسترسی
        items_to_keep = sorted(
            self.knowledge_base.items(),
            key=lambda x: (
                x[1]['metadata']['importance_score'],
                x[1]['metadata']['access_count']
            ),
            reverse=True
        )[:max_items]
        
        self.knowledge_base = dict(items_to_keep)
        self.memory_stats['total_knowledge_items'] = len(self.knowledge_base)
        self.memory_stats['memory_usage_mb'] = self._calculate_memory_usage()
        
        print(f"🧹 Memory cleaned up. Kept {len(self.knowledge_base)} items.")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """آمار پایگاه دانش"""
        categories = {}
        for item in self.knowledge_base.values():
            category = item['metadata']['category']
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'memory_stats': self.memory_stats,
            'categories_distribution': categories,
            'top_accessed': sorted(
                self.knowledge_base.items(),
                key=lambda x: x[1]['metadata']['access_count'],
                reverse=True
            )[:10],
            'most_important': sorted(
                self.knowledge_base.items(),
                key=lambda x: x[1]['metadata']['importance_score'],
                reverse=True
            )[:10],
            'system_health': {
                'memory_efficiency': self.memory_stats['memory_usage_mb'] / max(1, self.memory_stats['total_knowledge_items']),
                'access_rate': self.memory_stats['access_count'] / max(1, self.memory_stats['total_knowledge_items']),
                'category_diversity': len(categories)
            }
        }

# نمونه گلوبال
ai_memory = AIMemoryManager()
