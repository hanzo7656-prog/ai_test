# 📁 test_data_sources.py

from src.data.data_manager import SmartDataManager

def test_smart_data_manager():
    """تست مدیر داده هوشمند"""
    print("🧪 Testing Smart Data Manager...")
    
    # ایجاد مدیر داده
    data_manager = SmartDataManager()
    
    # تست دریافت داده کوین‌ها
    coins_data = data_manager.get_coins_data(limit=10)
    if coins_data:
        print(f"✅ Got {len(coins_data.get('result', []))} coins")
    else:
        print("❌ Failed to get coins data")
    
    # تست دریافت جزئیات بیت‌کوین
    btc_data = data_manager.get_coin_details("bitcoin")
    if btc_data:
        print(f"✅ Bitcoin price: ${btc_data.get('price', 'N/A')}")
    
    # نمایش آمار
    stats = data_manager.get_source_stats()
    print("📊 Data Source Stats:", stats)

if __name__ == "__main__":
    test_smart_data_manager()
