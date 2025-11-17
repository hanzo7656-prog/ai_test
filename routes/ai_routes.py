from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from datetime import datetime
import numpy as np
from typing import List, Optional

# ایمپورت کامپوننت‌های هوش مصنوعی
try:
    from simple_ai.brain import ai_brain
    from simple_ai.learner import ai_learner
    from simple_ai.memory import ai_memory
    from simple_ai.trainer import create_training_manager
    from integrations.data_connector import ai_data_connector
    from integrations.ai_monitor import ai_monitor
    
    # ایجاد Training Manager
    ai_trainer = create_training_manager(ai_brain, ai_learner, ai_memory)
    
    AI_SYSTEM_AVAILABLE = True
    print("✅ All AI components imported successfully!")
    
except ImportError as e:
    print(f"❌ AI system import failed: {e}")
    AI_SYSTEM_AVAILABLE = False

router = APIRouter(prefix="/ai", tags=["Artificial Intelligence"])

@router.get("/health")
async def ai_health():
    """بررسی سلامت کامل سیستم هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        # جمع‌آوری سلامت از تمام کامپوننت‌ها
        brain_health = ai_brain.get_network_health()
        learning_stats = ai_learner.get_learning_stats()
        memory_stats = ai_memory.get_knowledge_base_stats()
        connection_stats = ai_data_connector.get_connection_stats()
        training_report = ai_trainer.get_training_report()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "neural_network": {
                    "status": "active",
                    "neurons": brain_health['neuron_count'],
                    "active_neurons": brain_health['active_neurons'],
                    "accuracy": brain_health['performance']['current_accuracy']
                },
                "learning_engine": {
                    "status": "active",
                    "processed_files": learning_stats['processed_files'],
                    "vocabulary_size": learning_stats['vocabulary_size'],
                    "patterns_detected": learning_stats['learning_stats']['patterns_detected']
                },
                "memory_system": {
                    "status": "active",
                    "knowledge_items": memory_stats['memory_stats']['total_knowledge_items'],
                    "memory_usage_mb": memory_stats['memory_stats']['memory_usage_mb']
                },
                "data_connector": {
                    "status": connection_stats['status'],
                    "success_rate": connection_stats['success_rate_percent'],
                    "total_requests": connection_stats['connection_stats']['total_requests']
                }
            },
            "performance_summary": {
                "training_sessions": training_report['performance_metrics']['total_training_sessions'],
                "total_samples": training_report['performance_metrics']['total_training_samples'],
                "average_accuracy": training_report['performance_metrics']['average_accuracy'],
                "best_accuracy": training_report['performance_metrics']['best_accuracy']
            },
            "recommendations": training_report['recommendations']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI health check failed: {e}")

@router.post("/upload-training")
async def upload_training_file(file: UploadFile = File(...)):
    """آپلود فایل متنی برای آموزش هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, "Only .txt files are supported")
    
    try:
        # خواندن محتوای فایل
        content = await file.read()
        text_data = content.decode('utf-8')
        
        if len(text_data) < 10:
            raise HTTPException(400, "File content is too short")
        
        # آموزش با فایل آپلود شده
        training_result = ai_trainer.train_batch([text_data])
        
        # استخراج الگوها از متن
        patterns = ai_learner.extract_patterns(text_data)
        
        return {
            "status": "success",
            "filename": file.filename,
            "file_size_chars": len(text_data),
            "training_result": training_result,
            "patterns_detected": patterns,
            "learning_stats": ai_learner.get_learning_stats()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

@router.get("/predict")
async def ai_predict(text: str = Query(..., description="متن برای پیش‌بینی")):
    """پیش‌بینی هوش مصنوعی روی متن ورودی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        # پردازش متن و تبدیل به بردار
        input_vector = ai_learner.process_text_file(text)
        
        # فعال‌سازی شبکه برای پیش‌بینی
        prediction = ai_brain.activate(input_vector)
        
        # استخراج الگوها از متن
        patterns = ai_learner.extract_patterns(text)
        
        # محاسبه اطمینان پیش‌بینی
        confidence = float(np.mean(prediction))
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "input_text": text[:500] + "..." if len(text) > 500 else text,
            "prediction_confidence": confidence,
            "prediction_vector_size": len(prediction),
            "active_neurons": int(np.sum(prediction != 0)),
            "patterns_detected": patterns,
            "interpretation": self._interpret_prediction(confidence, patterns)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@router.get("/knowledge/stats")
async def knowledge_stats():
    """آمار دانش یادگرفته شده توسط هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        memory_stats = ai_memory.get_knowledge_base_stats()
        learning_stats = ai_learner.get_learning_stats()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "knowledge_base": memory_stats,
            "learning_progress": learning_stats,
            "summary": {
                "total_knowledge_items": memory_stats['memory_stats']['total_knowledge_items'],
                "vocabulary_size": learning_stats['vocabulary_size'],
                "patterns_learned": learning_stats['learning_stats']['patterns_detected'],
                "memory_usage_mb": memory_stats['memory_stats']['memory_usage_mb'],
                "system_health_score": self._calculate_knowledge_health_score(memory_stats, learning_stats)
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Knowledge stats failed: {str(e)}")

@router.post("/train/raw-data")
async def train_from_raw_data(
    source: str = Query(..., description="منبع داده: coins, news, insights, exchanges"),
    limit: int = Query(10, description="تعداد نمونه‌های آموزشی")
):
    """آموزش هوش مصنوعی از داده‌های خام سیستم"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        # دریافت داده‌های خام بر اساس منبع
        if source == "coins":
            raw_data = ai_data_connector.get_raw_coins_data(limit)
            training_texts = [f"ارز {item['name']} با قیمت {item['price_usd']} دلار و تغییر {item['change_24h']} درصد" for item in raw_data]
        elif source == "news":
            raw_data = ai_data_connector.get_raw_news_data(limit)
            training_texts = [f"خبر: {item['title']}. {item['content']}" for item in raw_data]
        elif source == "insights":
            raw_data = ai_data_connector.get_raw_insights_data(limit)
            training_texts = [f"تحلیل: {item['analysis']}. توصیه: {item['recommendation']}" for item in raw_data]
        elif source == "exchanges":
            raw_data = ai_data_connector.get_raw_exchanges_data(limit)
            training_texts = [f"صرافی {item['name']}: {item['details']}" for item in raw_data]
        else:
            raise HTTPException(400, "Invalid source. Use: coins, news, insights, exchanges")
        
        if not training_texts:
            raise HTTPException(404, f"No data available from {source}")
        
        # آموزش با داده‌های دریافتی
        training_result = ai_trainer.train_batch(training_texts)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "samples_used": len(training_texts),
            "training_result": training_result,
            "raw_data_sample": raw_data[:2] if raw_data else [],  # نمونه‌ای از داده‌ها
            "connection_stats": ai_data_connector.get_connection_stats()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Training from raw data failed: {str(e)}")

@router.get("/training/report")
async def training_report():
    """گزارش کامل آموزش هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        report = ai_trainer.get_training_report()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "training_report": report
        }
        
    except Exception as e:
        raise HTTPException(500, f"Training report failed: {str(e)}")

@router.post("/optimize")
async def optimize_ai():
    """بهینه‌سازی خودکار هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        # بهینه‌سازی پارامترها
        optimization_result = ai_trainer.adjust_hyperparameters()
        
        # بهینه‌سازی معماری شبکه
        ai_brain.optimize_architecture()
        
        # پاک‌سازی حافظه اگر لازم باشد
        ai_memory.cleanup_memory()
        
        return {
            "status": "optimized",
            "timestamp": datetime.now().isoformat(),
            "optimization_result": optimization_result,
            "architecture_optimized": True,
            "memory_cleaned": True,
            "current_network_health": ai_brain.get_network_health()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Optimization failed: {str(e)}")

# توابع کمکی
def _interpret_prediction(confidence: float, patterns: Dict) -> str:
    """تفسیر نتیجه پیش‌بینی"""
    if confidence > 0.8:
        return "پیش‌بینی با اطمینان بالا - الگوهای آشنا شناسایی شد"
    elif confidence > 0.6:
        return "پیش‌بینی با اطمینان متوسط - برخی الگوها شناسایی شد"
    elif confidence > 0.4:
        return "پیش‌بینی با اطمینان پایین - الگوهای محدودی شناسایی شد"
    else:
        return "پیش‌بینی با اطمینان بسیار پایین - الگوهای جدید یا ناشناخته"

def _calculate_knowledge_health_score(memory_stats: Dict, learning_stats: Dict) -> float:
    """محاسبه امتیاز سلامت دانش"""
    base_score = 50
    
    # امتیاز بر اساس حجم دانش
    knowledge_items = memory_stats['memory_stats']['total_knowledge_items']
    if knowledge_items > 50:
        base_score += 20
    elif knowledge_items > 20:
        base_score += 10
    
    # امتیاز بر اساس دایره واژگان
    vocab_size = learning_stats['vocabulary_size']
    if vocab_size > 100:
        base_score += 15
    elif vocab_size > 50:
        base_score += 10
    
    # امتیاز بر اساس الگوها
    patterns = learning_stats['learning_stats']['patterns_detected']
    if patterns > 20:
        base_score += 15
    elif patterns > 10:
        base_score += 10
    
    return min(base_score, 100)
