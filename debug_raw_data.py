# debug_raw_data.py
from complete_coinstats_manager import coin_stats_manager
import json

def debug_raw_structure():
    print("🔍 بررسی ساختار داده‌های خام CoinStats...")
    
    try:
        # تست ۱: لیست کوین‌ها
        print("\n1. تست لیست کوین‌ها:")
        coins = coin_stats_manager.get_coins_list(limit=3)
        print("✅ ساختار کلی:", list(coins.keys()) if coins else "خطا")
        if coins and 'result' in coins:
            print(f"   تعداد کوین‌ها: {len(coins['result'])}")
            if coins['result']:
                first_coin = coins['result'][0]
                print(f"   فیلدهای اولین کوین: {list(first_coin.keys())[:10]}...")
                print(f"   نمونه داده: { {k: first_coin[k] for k in list(first_coin.keys())[:5]} }")
        
        # تست ۲: جزئیات بیت‌کوین
        print("\n2. تست جزئیات بیت‌کوین:")
        btc = coin_stats_manager.get_coin_details("bitcoin", "USD")
        print("✅ ساختار کلی:", list(btc.keys()) if btc else "خطا")
        if btc and 'result' in btc:
            btc_data = btc['result']
            print(f"   فیلدهای BTC: {list(btc_data.keys())[:15]}...")
            # نمایش مهم‌ترین فیلدها
            important_fields = ['price', 'priceChange1d', 'volume', 'marketCap', 'high', 'low', 'rank']
            for field in important_fields:
                print(f"   {field}: {btc_data.get(field, 'NOT_FOUND')}")
        
        # تست ۳: چارت‌ها
        print("\n3. تست داده‌های چارت:")
        charts = coin_stats_manager.get_coin_charts("bitcoin", "1w")
        print("✅ ساختار کلی:", list(charts.keys()) if charts else "خطا")
        if charts and 'result' in charts:
            chart_data = charts['result']
            print(f"   تعداد نقاط چارت: {len(chart_data)}")
            if chart_data:
                print(f"   نمونه نقطه چارت: {chart_data[0]}")
        
        print("\n🎯 نتیجه‌گیری: داده‌ها به صورت خام دریافت می‌شوند و نیاز به پردازش دارند")
        
    except Exception as e:
        print(f"❌ خطا در دیباگ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_raw_data_structure()
