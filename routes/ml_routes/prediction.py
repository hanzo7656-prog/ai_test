# routes/ml_routes/predictions.py
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import torch

from ml_core import ml_model_manager, performance_tracker
from data_pipeline import feature_engineer

logger = logging.getLogger(__name__)

predictions_router = APIRouter(prefix="/api/ml/predict", tags=["ML Predictions"])

@predictions_router.post("/technical/{model_name}")
async def predict_technical(
    model_name: str,
    prediction_request: Dict[str, Any] = Body(...)
):
    """دریافت پیش‌بینی از مدل تحلیل تکنیکال"""
    try:
        # بررسی وجود مدل
        if model_name not in ml_model_manager.active_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        logger.info(f"🎯 Making prediction with {model_name}")
        
        # آماده‌سازی داده ورودی
        input_data = await _prepare_prediction_input(prediction_request, model_name)
        
        # اجرای پیش‌بینی
        start_time = datetime.now()
        prediction_result = await ml_model_manager.predict(model_name, input_data)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # ردیابی عملکرد
        performance_tracker.track_inference(
            model_name=model_name,
            inference_time=inference_time,
            confidence=prediction_result.get('confidence', 0.5),
            success=True,
            input_size=input_data.shape
        )
        
        # ساخت پاسخ
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "inference_time_seconds": inference_time,
            "prediction": prediction_result
        }
        
        logger.info(f"✅ Prediction completed in {inference_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction failed for {model_name}: {e}")
        
        # ردیابی خطا
        if 'performance_tracker' in locals():
            performance_tracker.track_inference(
                model_name=model_name,
                inference_time=0,
                confidence=0,
                success=False,
                input_size=(0,)
            )
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@predictions_router.post("/batch/technical")
async def batch_predict_technical(
    batch_request: Dict[str, Any] = Body(...)
):
    """پیش‌بینی دسته‌ای برای داده‌های متعدد"""
    try:
        model_name = batch_request.get('model_name', 'technical_analyzer')
        
        if model_name not in ml_model_manager.active_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        input_data_list = batch_request.get('data', [])
        
        if not input_data_list:
            raise HTTPException(status_code=400, detail="No data provided for batch prediction")
        
        logger.info(f"🎯 Starting batch prediction with {model_name} for {len(input_data_list)} items")
        
        # آماده‌سازی داده‌های دسته‌ای
        batch_tensors = []
        for i, data_item in enumerate(input_data_list):
            try:
                input_tensor = await _prepare_prediction_input(data_item, model_name)
                batch_tensors.append(input_tensor)
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare item {i}: {e}")
                continue
        
        if not batch_tensors:
            raise HTTPException(status_code=400, detail="No valid data items found")
        
        # اجرای پیش‌بینی‌های دسته‌ای
        start_time = datetime.now()
        batch_results = await ml_model_manager.batch_predict(model_name, batch_tensors)
        total_time = (datetime.now() - start_time).total_seconds()
        
        # ساخت پاسخ
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "total_processing_time": total_time,
            "average_time_per_prediction": total_time / len(batch_results),
            "successful_predictions": len(batch_results),
            "failed_predictions": len(input_data_list) - len(batch_results),
            "predictions": batch_results
        }
        
        logger.info(f"✅ Batch prediction completed: {len(batch_results)} successful predictions")
        return response
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@predictions_router.get("/confidence/{model_name}")
async def get_model_confidence(model_name: str):
    """دریافت سطح اطمینان فعلی مدل"""
    try:
        if model_name not in ml_model_manager.active_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # دریافت متریک‌های عملکرد
        from ml_core import performance_tracker
        performance_data = performance_tracker.get_model_performance(model_name, "1h")
        
        confidence_data = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "current_confidence": performance_data.get('quality_metrics', {}).get('avg_confidence', 0),
            "confidence_trend": "stable",  # می‌تواند از تاریخچه محاسبه شود
            "performance_metrics": {
                "success_rate": performance_data.get('summary', {}).get('success_rate', 0),
                "recent_inferences": performance_data.get('summary', {}).get('total_inferences', 0)
            }
        }
        
        return {
            "status": "success",
            "data": confidence_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting model confidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@predictions_router.post("/market/sentiment")
async def predict_market_sentiment(
    sentiment_request: Dict[str, Any] = Body(...)
):
    """پیش‌بینی احساسات بازار (placeholder برای مدل آینده)"""
    try:
        # این یک پیاده‌سازی placeholder است
        # وقتی مدل تحلیل احساسات اضافه شد، کامل می‌شود
        
        news_data = sentiment_request.get('news_data', [])
        market_data = sentiment_request.get('market_data', {})
        
        # تحلیل ساده احساسات
        sentiment_score = _calculate_simple_sentiment(news_data, market_data)
        
        prediction_result = {
            "sentiment_score": sentiment_score,
            "sentiment_label": "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral",
            "confidence": 0.75,  # placeholder
            "factors_considered": {
                "news_count": len(news_data),
                "market_trend": market_data.get('trend', 'unknown'),
                "analysis_method": "simple_heuristic"
            }
        }
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": prediction_result
        }
        
    except Exception as e:
        logger.error(f"❌ Market sentiment prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@predictions_router.post("/custom/{model_name}")
async def custom_model_prediction(
    model_name: str,
    custom_request: Dict[str, Any] = Body(...)
):
    """پیش‌بینی با مدل سفارشی"""
    try:
        if model_name not in ml_model_manager.active_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # دریافت داده ورودی از درخواست
        input_data = custom_request.get('input_data')
        if input_data is None:
            raise HTTPException(status_code=400, detail="input_data is required")
        
        # تبدیل به تانسور (بسته به فرمت داده)
        input_tensor = _convert_custom_input(input_data, model_name)
        
        # اجرای پیش‌بینی
        prediction_result = await ml_model_manager.predict(model_name, input_tensor)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prediction": prediction_result
        }
        
    except Exception as e:
        logger.error(f"❌ Custom prediction failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _prepare_prediction_input(prediction_data: Dict[str, Any], model_name: str) -> torch.Tensor:
    """آماده‌سازی داده ورودی برای پیش‌بینی"""
    try:
        if model_name == 'technical_analyzer':
            # برای تحلیل‌گر تکنیکال، از ویژگی‌های مهندسی شده استفاده می‌کنیم
            if 'raw_data' in prediction_data:
                # اگر داده خام داریم، ویژگی‌های جدید تولید کنیم
                from data_pipeline import feature_engineer
                engineered_features = feature_engineer.engineer_market_features({
                    'sources': {'custom_data': {'status': 'success', 'data': prediction_data['raw_data']}}
                })
            else:
                # اگر ویژگی‌های از قبل مهندسی شده داریم
                engineered_features = prediction_data.get('engineered_features', {})
            
            # تبدیل به تانسور
            return await _features_to_tensor(engineered_features)
            
        else:
            # برای مدل‌های دیگر، منطق متفاوت
            raise ValueError(f"Model {model_name} not supported yet")
            
    except Exception as e:
        logger.error(f"❌ Error preparing prediction input: {e}")
        raise

async def _features_to_tensor(engineered_features: Dict[str, Any]) -> torch.Tensor:
    """تبدیل ویژگی‌های مهندسی شده به تانسور"""
    import numpy as np
    
    # استخراج تمام مقادیر عددی
    numeric_values = []
    
    def extract_numeric_values(data):
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                elif isinstance(value, dict):
                    extract_numeric_values(value)
                elif isinstance(value, list):
                    for item in value[:3]:  # فقط 3 آیتم اول لیست
                        if isinstance(item, (int, float)):
                            numeric_values.append(item)
    
    extract_numeric_values(engineered_features)
    
    if not numeric_values:
        raise ValueError("No numeric values found in features")
    
    # نرمال‌سازی و تبدیل به تانسور
    values_array = np.array(numeric_values, dtype=np.float32)
    
    # اگر تعداد ویژگی‌ها کم است، padding کن
    if len(values_array) < 50:
        values_array = np.pad(values_array, (0, 50 - len(values_array)), 'constant')
    elif len(values_array) > 100:  # اگر زیاد است، نمونه‌گیری کن
        values_array = values_array[:100]
    
    # تبدیل به شکل مورد نیاز (batch_size, sequence_length, features)
    input_tensor = torch.FloatTensor(values_array).unsqueeze(0).unsqueeze(0)
    return input_tensor

def _calculate_simple_sentiment(news_data: List[Dict], market_data: Dict) -> float:
    """محاسبه ساده احساسات بازار"""
    if not news_data:
        return 0.5  # خنثی
    
    positive_keywords = ['صعود', 'رشد', 'سود', 'مثبت', 'قوی', 'بهبود', 'خرید']
    negative_keywords = ['نزول', 'سقوط', 'ضرر', 'منفی', 'ضعیف', 'ریزش', 'فروش']
    
    sentiment_score = 0.5
    keyword_weight = 0.1
    
    for news_item in news_data:
        text = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        total_keywords = positive_count + negative_count
        if total_keywords > 0:
            item_sentiment = positive_count / total_keywords
            # وزن بر اساس تازگی خبر
            sentiment_score = (sentiment_score + item_sentiment * keyword_weight) / (1 + keyword_weight)
    
    return max(0.0, min(1.0, sentiment_score))

def _convert_custom_input(input_data: Any, model_name: str) -> torch.Tensor:
    """تبدیل داده سفارشی به تانسور"""
    # این تابع بسته به نوع مدل و فرمت داده می‌تواند متفاوت باشد
    if isinstance(input_data, list):
        return torch.FloatTensor(input_data)
    elif isinstance(input_data, dict):
        # تبدیل دیکشنری به لیست مقادیر
        values = list(input_data.values())
        return torch.FloatTensor(values)
    else:
        raise ValueError(f"Unsupported input data type: {type(input_data)}")
