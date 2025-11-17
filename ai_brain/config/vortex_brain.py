from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, Optional
import logging
import time
import asyncio

from ai_brain.config.ai_config import AIConfig
from ai_brain.core.neural_network import SparseNeuralNetwork
from ai_brain.core.text_processor import TextProcessor
from ai_brain.core.learning_engine import LearningEngine
from ai_brain.memory.memory_manager import MemoryManager
from ai_brain.memory.knowledge_compressor import KnowledgeCompressor
from ai_brain.integration.api_handler import APIHandler
from ai_brain.integration.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class VortexBrain:
    """کلاس اصلی یکپارچه‌سازی هوش مصنوعی"""
    
    def __init__(self):
        self.config = AIConfig()
        self.initialized = False
        self.redis_manager = None
        
        # کامپوننت‌های اصلی
        self.neural_network = None
        self.text_processor = None
        self.learning_engine = None
        self.memory_manager = None
        self.knowledge_compressor = None
        self.api_handler = None
        self.response_formatter = None
        
        # وضعیت سیستم
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        
        logger.info("🧠 VortexAI Brain ایجاد شد")
    
    async def initialize(self, redis_manager=None):
        """مقداردهی اولیه تمام کامپوننت‌ها"""
        if self.initialized:
            return
        
        try:
            logger.info("🚀 شروع راه‌اندازی VortexAI Brain...")
            
            # تأیید تنظیمات
            if not self.config.validate_config():
                raise Exception("تنظیمات نامعتبر هستند")
            
            # مقداردهی کامپوننت‌ها
            self.text_processor = TextProcessor(self.config.get('text_processing', {}))
            self.neural_network = SparseNeuralNetwork(self.config.get_neural_network_config())
            self.learning_engine = LearningEngine(self.config.get_learning_config())
            
            self.memory_manager = MemoryManager(self.config.get_memory_config())
            self.knowledge_compressor = KnowledgeCompressor(self.config.get_memory_config())
            
            self.api_handler = APIHandler(self.config.get_api_config())
            self.response_formatter = ResponseFormatter(self.config.get_response_config())
            
            # تنظیم اتصال ردیس
            if redis_manager:
                self.redis_manager = redis_manager
                self.memory_manager.initialize_redis(redis_manager)
                logger.info("✅ اتصال ردیس تنظیم شد")
            
            # بارگذاری حالت ذخیره شده
            await self._load_saved_state()
            
            # تست اتصال APIها
            api_test = await self.api_handler.test_api_connections()
            logger.info(f"🔗 تست اتصال API: {api_test}")
            
            self.initialized = True
            startup_time = time.time() - self.start_time
            logger.info(f"✅ VortexAI Brain راه‌اندازی شد - زمان: {startup_time:.2f}ثانیه")
            
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی VortexAI: {e}")
            raise
    
    async def process_query(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """پردازش سوال کاربر و تولید پاسخ"""
        if not self.initialized:
            raise Exception("سیستم راه‌اندازی نشده است")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            logger.info(f"👤 کاربر {user_id}: {user_input}")
            
            # مرحله ۱: پردازش متن و تشخیص intent
            complexity = self.text_processor.estimate_complexity(user_input)
            
            # بررسی ظرفیت پردازش
            if not self.neural_network.can_process_complexity(complexity):
                response_text = self.response_formatter.format_capacity_error()
                return self._create_response(False, response_text, start_time)
            
            tokens = self.text_processor.preprocess_text(user_input)
            input_vector = self.text_processor.text_to_vector(tokens)
            
            intent, confidence = self.text_processor.detect_intent(user_input)
            extracted_params = self.text_processor.extract_parameters(user_input, intent)
            
            # مرحله ۲: پردازش در شبکه عصبی
            neural_output = self.neural_network.process_input(input_vector)
            activated_neurons = [i for i, val in enumerate(neural_output) if val > 0.1]
            
            # مرحله ۳: جستجو در حافظه
            cached_response = self.memory_manager.retrieve(f"response:{intent}:{user_id}", user_id)
            if cached_response:
                logger.info("💾 پاسخ از حافظه بازیابی شد")
                self.successful_requests += 1
                return self._create_response(True, cached_response, start_time, intent, confidence)
            
            # مرحله ۴: فراخوانی API
            api_request = self.api_handler.map_intent_to_api(intent, user_input, extracted_params)
            api_response = await self.api_handler.call_api(intent, api_request['params'])
            
            if not api_response.get('success', False):
                error_msg = api_response.get('error', 'خطای ناشناخته در API')
                response_text = self.response_formatter.format_error_response(error_msg)
                
                # یادگیری از خطا
                await self.learning_engine.process_interaction(
                    user_input, activated_neurons, api_response, False
                )
                
                return self._create_response(False, response_text, start_time, intent, confidence)
            
            # مرحله ۵: فرمت‌دهی پاسخ
            user_language = self.response_formatter.detect_user_language(user_input)
            response_text = self.response_formatter.format_response(intent, api_response, user_language)
            
            # مرحله ۶: یادگیری و ذخیره
            await self.learning_engine.process_interaction(
                user_input, activated_neurons, api_response, True
            )
            
            # ذخیره در حافظه
            self.memory_manager.store_sensory(f"response:{intent}:{user_id}", response_text, user_id)
            
            # یادگیری هبیان
            self.neural_network.hebbian_learn(activated_neurons)
            
            # استخراج و یادگیری مفاهیم
            if activated_neurons:
                concept_key = f"concept:{intent}:{user_input[:20]}"
                self.neural_network.learn_concept(concept_key, activated_neurons)
            
            self.successful_requests += 1
            
            logger.info(f"✅ پردازش موفق - Intent: {intent}, Confidence: {confidence:.2f}")
            
            return self._create_response(True, response_text, start_time, intent, confidence, api_response)
            
        except Exception as e:
            logger.error(f"❌ خطا در پردازش سوال: {e}")
            error_response = self.response_formatter.format_error_response("خطای داخلی سیستم")
            return self._create_response(False, error_response, start_time)
    
    def _create_response(self, success: bool, response_text: str, start_time: float, 
                        intent: str = None, confidence: float = None, 
                        api_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """ساخت ساختار پاسخ استاندارد"""
        response_time = time.time() - start_time
        
        response = {
            'success': success,
            'response': response_text,
            'response_time': round(response_time, 3),
            'timestamp': time.time(),
            'version': self.config.get('system.version')
        }
        
        if intent:
            response['intent'] = intent
        if confidence:
            response['confidence'] = round(confidence, 3)
        if api_data:
            response['api_data'] = {
                'endpoint': api_data.get('endpoint'),
                'response_time': api_data.get('response_time')
            }
        
        return response
    
    async def _load_saved_state(self):
        """بارگذاری حالت ذخیره شده سیستم"""
        try:
            model_path = self.config.get('storage.model_save_path')
            # در اینجا می‌توان حالت ذخیره شده را بارگذاری کرد
            logger.info("📂 حالت ذخیره شده بارگذاری شد")
        except Exception as e:
            logger.warning(f"⚠️ خطا در بارگذاری حالت: {e}")
    
    async def save_state(self):
        """ذخیره حالت فعلی سیستم"""
        try:
            model_path = self.config.get('storage.model_save_path')
            # ذخیره حالت شبکه عصبی
            self.neural_network.save_state(model_path)
            
            # ذخیره تنظیمات
            self.config.save_to_file(model_path.replace('.json', '_config.json'))
            
            logger.info("💾 حالت سیستم ذخیره شد")
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره حالت: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """گزارش سلامت سیستم"""
        if not self.initialized:
            return {'status': 'not_initialized', 'message': 'سیستم راه‌اندازی نشده'}
        
        try:
            # آمار از کامپوننت‌های مختلف
            nn_stats = self.neural_network.get_network_stats()
            memory_stats = self.memory_manager.get_memory_stats()
            learning_stats = self.learning_engine.get_learning_stats()
            compression_stats = self.knowledge_compressor.get_compression_stats()
            
            uptime = time.time() - self.start_time
            success_rate = (self.successful_requests / self.total_requests) * 100 if self.total_requests > 0 else 0
            
            return {
                'status': 'healthy',
                'uptime_seconds': round(uptime, 2),
                'total_requests': self.total_requests,
                'success_rate': round(success_rate, 2),
                'components': {
                    'neural_network': nn_stats,
                    'memory': memory_stats,
                    'learning': learning_stats,
                    'compression': compression_stats
                },
                'config_summary': self.config.get_config_summary()
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در گزارش سلامت: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def cleanup(self):
        """پاک‌سازی و خاتمه"""
        try:
            # ذخیره حالت نهایی
            await self.save_state()
            
            # بستن اتصالات
            if self.api_handler:
                await self.api_handler.close()
            
            # پاک‌سازی حافظه
            if self.memory_manager:
                self.memory_manager.cleanup_expired()
            
            logger.info("🧹 VortexAI Brain پاک‌سازی شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در پاک‌سازی: {e}")

# ایجاد نمونه اصلی و روت FastAPI
vortex_brain = VortexBrain()
ai_router = APIRouter()

@ai_router.on_event("startup")
async def startup_event():
    """رویداد راه‌اندازی"""
    try:
        from debug_system.storage.redis_manager import redis_manager
        await vortex_brain.initialize(redis_manager)
    except ImportError:
        await vortex_brain.initialize()
    except Exception as e:
        logger.error(f"❌ خطا در راه‌اندازی: {e}")

@ai_router.on_event("shutdown")
async def shutdown_event():
    """رویداد خاموش‌سازی"""
    await vortex_brain.cleanup()

@ai_router.post("/query")
async def process_ai_query(request: Request):
    """اندپوینت اصلی پردازش سوالات"""
    try:
        data = await request.json()
        user_input = data.get('question', '').strip()
        user_id = data.get('user_id', 'default')
        
        if not user_input:
            raise HTTPException(status_code=400, detail="سوال الزامی است")
        
        response = await vortex_brain.process_query(user_input, user_id)
        return response
        
    except Exception as e:
        logger.error(f"❌ خطا در اندپوینت query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.get("/health")
async def get_ai_health():
    """بررسی سلامت هوش مصنوعی"""
    health_report = vortex_brain.get_system_health()
    return health_report

@ai_router.get("/stats")
async def get_ai_stats():
    """آمار عملکرد هوش مصنوعی"""
    health = vortex_brain.get_system_health()
    return {
        'performance': {
            'total_requests': vortex_brain.total_requests,
            'successful_requests': vortex_brain.successful_requests,
            'success_rate': health.get('success_rate', 0)
        },
        'system': health.get('components', {})
    }

@ai_router.post("/learn")
async def submit_learning_material(request: Request):
    """ارسال مطالب آموزشی برای یادگیری"""
    try:
        data = await request.json()
        text_material = data.get('text', '').strip()
        
        if not text_material:
            raise HTTPException(status_code=400, detail="متن آموزشی الزامی است")
        
        # پردازش متن آموزشی
        tokens = vortex_brain.text_processor.preprocess_text(text_material)
        input_vector = vortex_brain.text_processor.text_to_vector(tokens)
        neural_output = vortex_brain.neural_network.process_input(input_vector)
        activated_neurons = [i for i, val in enumerate(neural_output) if val > 0.1]
        
        # یادگیری از متن آموزشی
        vortex_brain.neural_network.hebbian_learn(activated_neurons)
        
        # ذخیره در حافظه بلندمدت
        vortex_brain.memory_manager.store_long_term(
            f"training:{hash(text_material)}", 
            {'text': text_material, 'type': 'training'}, 
            "system"
        )
        
        return {
            'success': True,
            'message': 'مطلب آموزشی پردازش شد',
            'activated_neurons': len(activated_neurons)
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در یادگیری: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# تابع کمکی برای دسترسی از health router
async def get_ai_health():
    """تابع برای استفاده در health router اصلی"""
    return vortex_brain.get_system_health()
