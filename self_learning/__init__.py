# self_learning/__init__.py
"""
سیستم یادگیری خودکار - Self Learning Module
ماژول‌های آموزش خودکار، یادگیری تقویتی و مدیریت دانش

کامپوننت‌های اصلی:
1. 🎓 Autonomous Trainer - آموزش خودکار مدل‌ها
2. 🤖 Reinforcement Learner - یادگیری تقویتی پیشرفته  
3. 📚 Knowledge Base - پایگاه دانش و حافظه مدل‌ها

ویژگی‌های کلیدی:
- یادگیری پیوسته و تطبیقی
- بهینه‌سازی خودکار hyperparameters
- مدیریت دانش و تجربیات
- انتقال یادگیری بین مدل‌ها
"""

from .autonomous_trainer import autonomous_trainer, AutonomousTrainer, initialize_autonomous_trainer
from .reinforcement_learner import reinforcement_learner, ReinforcementLearner
from .knowledge_base import knowledge_base, KnowledgeBase

__all__ = [
    # کلاس‌های اصلی
    'AutonomousTrainer',
    'ReinforcementLearner', 
    'KnowledgeBase',
    
    # نمونه‌های global
    'autonomous_trainer',
    'reinforcement_learner',
    'knowledge_base',
    
    # توابع مقداردهی
    'initialize_autonomous_trainer'
]

# اطلاعات ماژول
__version__ = "1.0.0"
__author__ = "Vortex AI System"
__description__ = "سیستم یادگیری خودکار و پیشرفته هوش مصنوعی"

def get_self_learning_info():
    """دریافت اطلاعات کامل ماژول Self Learning"""
    return {
        "module": "self_learning",
        "version": __version__,
        "description": __description__,
        "components": {
            "autonomous_trainer": {
                "status": "ready",
                "features": [
                    "Continuous learning loop",
                    "Automatic training scheduling", 
                    "Performance-based training triggers",
                    "Multi-model training management"
                ]
            },
            "reinforcement_learner": {
                "status": "ready", 
                "features": [
                    "Q-learning and policy gradients",
                    "Reward shaping and optimization",
                    "Experience replay buffer",
                    "Multi-agent learning support"
                ]
            },
            "knowledge_base": {
                "status": "ready",
                "features": [
                    "Model experience storage",
                    "Knowledge transfer between models",
                    "Learning pattern analysis",
                    "Performance history tracking"
                ]
            }
        },
        "capabilities": {
            "adaptive_learning": True,
            "knowledge_transfer": True,
            "automated_training": True,
            "reinforcement_learning": True,
            "performance_optimization": True
        }
    }

def initialize_self_learning(model_manager, data_integrator):
    """مقداردهی اولیه کامل ماژول Self Learning"""
    try:
        # مقداردهی Autonomous Trainer
        trainer = initialize_autonomous_trainer(model_manager, data_integrator)
        
        # مقداردهی Reinforcement Learner
        from .reinforcement_learner import initialize_reinforcement_learner
        rl_learner = initialize_reinforcement_learner(model_manager)
        
        # مقداردهی Knowledge Base
        from .knowledge_base import initialize_knowledge_base
        kb = initialize_knowledge_base()
        
        print("✅ Self Learning module initialized successfully!")
        print(f"   - Autonomous Trainer: Ready")
        print(f"   - Reinforcement Learner: Ready") 
        print(f"   - Knowledge Base: Ready")
        
        return {
            "autonomous_trainer": trainer,
            "reinforcement_learner": rl_learner,
            "knowledge_base": kb,
            "info": get_self_learning_info()
        }
        
    except Exception as e:
        print(f"❌ Error initializing Self Learning module: {e}")
        raise

# اجرای خودکار مقداردهی اولیه هنگام ایمپورت
try:
    # این بعداً وقتی model_manager و data_integrator موجود باشند پر می‌شود
    self_learning_initialized = None
    print("🤖 Self Learning module imported - call initialize_self_learning() to setup")
except Exception as e:
    print(f"⚠️ Self Learning auto-initialization skipped: {e}")
