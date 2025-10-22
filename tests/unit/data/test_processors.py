# 📁 test_processors.py

from src.data.data_manager import SmartDataManager
from src.data.processing_pipeline import DataProcessingPipeline

def test_processing_pipeline():
    """تست کامل پایپ‌لاین پردازش داده"""
    print("🧪 Testing Data Processing Pipeline...")
    
    # 1. دریافت داده خام
    data_manager = SmartDataManager()
    raw_data = data_manager.get_coins_data(limit=5)
    
    if not raw_data:
        print("❌ No raw data available")
        return
    
    print(f"📥 Raw data: {len(raw_data.get('result', []))} coins")
    
    # 2. پردازش داده
    pipeline = DataProcessingPipeline()
    processed_data = pipeline.process_raw_data(raw_data)
    
    if processed_data:
        print(f"📊 Processed data: {len(processed_data.get('result', []))} coins")
        
        # نمایش نمونه‌ای از داده پردازش‌شده
        sample_coin = processed_data['result'][0] if processed_data['result'] else {}
        print(f"🔍 Sample processed coin: {list(sample_coin.keys())[:5]}...")
        
        # نمایش آمار
        pipeline_info = pipeline.get_pipeline_info()
        print("📈 Pipeline Stats:", pipeline_info['pipeline_stats'])
        print("⚡ Performance:", pipeline_info['performance_summary'])
    else:
        print("❌ Data processing failed")

if __name__ == "__main__":
    test_processing_pipeline()
