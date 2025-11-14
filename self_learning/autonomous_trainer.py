# self_learning/autonomous_trainer.py
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

class AutonomousTrainer:
    """سیستم آموزش خودکار مدل‌های هوش مصنوعی"""
    
    def __init__(self, model_manager, data_integrator):
        self.model_manager = model_manager
        self.data_integrator = data_integrator
        self.training_schedule = {}
        self.training_history = []
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🎓 Autonomous Trainer initialized")

    async def continuous_training_loop(self):
        """حلقه آموزش پیوسته"""
        logger.info("🔄 Starting continuous training loop...")
        
        while True:
            try:
                # بررسی نیاز به آموزش
                training_needed = await self._check_training_need()
                
                if training_needed:
                    logger.info("📚 Training needed, starting training session...")
                    await self._conduct_training_session()
                
                # انتظار قبل از بررسی مجدد
                await asyncio.sleep(300)  # هر 5 دقیقه بررسی کن
                
            except Exception as e:
                logger.error(f"❌ Error in training loop: {e}")
                await asyncio.sleep(60)  # در صورت خطا 1 دقیقه صبر کن

    async def _check_training_need(self) -> bool:
        """بررسی نیاز به آموزش مدل‌ها"""
        try:
            # بررسی عملکرد مدل‌ها
            performance_report = self._get_performance_report()
            
            # بررسی داده‌های جدید
            data_report = await self.data_integrator.collect_raw_data()
            
            # شرایط نیاز به آموزش:
            # 1. کاهش performance
            # 2. داده‌های جدید کافی
            # 3. زمان از آموزش گذشته گذشته کافی
            needs_training = (
                self._has_performance_degradation(performance_report) and
                data_report['metadata']['successful_sources'] >= 2 and
                self._sufficient_time_since_last_training()
            )
            
            return needs_training
            
        except Exception as e:
            logger.error(f"❌ Error checking training need: {e}")
            return False

    def _get_performance_report(self) -> Dict[str, Any]:
        """دریافت گزارش عملکرد از performance tracker"""
        try:
            # اینجا باید با performance tracker ارتباط برقرار کنید
            # برای نمونه یک ساختار mock برمی‌گردانیم
            return {
                'models': {
                    'technical_analyzer': {
                        'success_rate': 0.92,
                        'avg_confidence': 0.85,
                        'trend': 'stable'
                    }
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting performance report: {e}")
            return {'models': {}}

    def _has_performance_degradation(self, performance_report: Dict[str, Any]) -> bool:
        """بررسی کاهش عملکرد مدل‌ها"""
        # پیاده‌سازی منطق تشخیص کاهش performance
        for model_name, metrics in performance_report.get('models', {}).items():
            success_rate = metrics.get('success_rate', 1.0)
            avg_confidence = metrics.get('avg_confidence', 1.0)
            
            if success_rate < 0.9 or avg_confidence < 0.8:
                return True
                
        return False

    def _sufficient_time_since_last_training(self) -> bool:
        """بررسی اینکه زمان کافی از آخرین آموزش گذشته باشد"""
        if not self.training_history:
            return True
            
        last_training = max([t['timestamp'] for t in self.training_history])
        last_training_time = datetime.fromisoformat(last_training)
        time_since_training = datetime.now() - last_training_time
        
        return time_since_training > timedelta(hours=4)  # حداقل 4 ساعت فاصله

    async def _conduct_training_session(self):
        """انجام یک جلسه آموزش"""
        training_session = {
            'session_id': f"train_{datetime.now().strftime('%Y%m%d_%H%M')}",
            'timestamp': datetime.now().isoformat(),
            'models_trained': [],
            'results': {},
            'status': 'started'
        }
        
        try:
            # جمع‌آوری داده‌های آموزشی
            training_data = await self.data_integrator.get_structured_training_data()
            
            if not training_data['training_ready']:
                logger.warning("⚠️ Training data not ready, skipping session")
                training_session['status'] = 'skipped'
                training_session['reason'] = 'insufficient_data'
                return
            
            # آموزش مدل‌های موجود
            for model_name in self.model_manager.active_models.keys():
                try:
                    result = await self._train_single_model(model_name, training_data)
                    training_session['models_trained'].append(model_name)
                    training_session['results'][model_name] = result
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_name}: {e}")
                    training_session['results'][model_name] = {'error': str(e)}
            
            training_session['status'] = 'completed'
            training_session['completion_time'] = datetime.now().isoformat()
            
            logger.info(f"✅ Training session completed: {len(training_session['models_trained'])} models trained")
            
        except Exception as e:
            logger.error(f"❌ Training session failed: {e}")
            training_session['status'] = 'failed'
            training_session['error'] = str(e)
        
        finally:
            # ذخیره تاریخچه آموزش
            self.training_history.append(training_session)
            
            # ذخیره در کش
            self.cache_manager.set_data(
                "utb", 
                f"training_session:{training_session['session_id']}", 
                training_session, 
                expire=86400
            )

    async def _train_single_model(self, model_name: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """آموزش یک مدل خاص"""
        training_result = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'training_data_quality': training_data['statistics']['data_quality'],
            'samples_used': training_data['statistics']['total_samples'],
            'improvement_metrics': {}
        }
        
        try:
            # اینجا منطق آموزش خاص هر مدل پیاده‌سازی می‌شود
            # برای نمونه، یک آموزش ساده شبیه‌سازی می‌کنیم
            
            if model_name == "technical_analyzer":
                result = await self._train_technical_analyzer(training_data)
                training_result.update(result)
            else:
                training_result['status'] = 'skipped'
                training_result['reason'] = 'no_training_logic'
            
            return training_result
            
        except Exception as e:
            logger.error(f"❌ Error in model training {model_name}: {e}")
            training_result['status'] = 'failed'
            training_result['error'] = str(e)
            return training_result

    async def _train_technical_analyzer(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """آموزش تحلیل‌گر تکنیکال"""
        # شبیه‌سازی آموزش
        await asyncio.sleep(2)  # شبیه‌سازی زمان آموزش
        
        return {
            'status': 'completed',
            'training_time': '2 seconds (simulated)',
            'improvement_metrics': {
                'accuracy_improvement': 0.02,
                'confidence_improvement': 0.03,
                'loss_reduction': 0.15
            },
            'new_metrics': {
                'accuracy': 0.94,
                'confidence': 0.88,
                'loss': 0.12
            }
        }

    def get_training_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """دریافت تاریخچه آموزش"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        return [
            session for session in self.training_history
            if datetime.fromisoformat(session['timestamp']) > cutoff_time
        ]

    def schedule_training(self, model_name: str, schedule: Dict[str, Any]):
        """زمان‌بندی آموزش برای مدل خاص"""
        self.training_schedule[model_name] = {
            'schedule': schedule,
            'last_trained': None,
            'next_training': self._calculate_next_training(schedule)
        }
        
        logger.info(f"📅 Training scheduled for {model_name}: {schedule}")

    def _calculate_next_training(self, schedule: Dict[str, Any]) -> datetime:
        """محاسبه زمان آموزش بعدی"""
        if schedule.get('interval') == 'daily':
            return datetime.now() + timedelta(days=1)
        elif schedule.get('interval') == 'weekly':
            return datetime.now() + timedelta(weeks=1)
        else:  # hourly
            return datetime.now() + timedelta(hours=schedule.get('hours', 4))

# نمونه global
autonomous_trainer = None

def initialize_autonomous_trainer(model_manager, data_integrator):
    """مقداردهی اولیه autonomous trainer"""
    global autonomous_trainer
    autonomous_trainer = AutonomousTrainer(model_manager, data_integrator)
    return autonomous_trainer
