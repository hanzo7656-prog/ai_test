# test_remote_data.py
from api_client import CoinStatsAPIClient

def test_remote_data():
    client = CoinStatsAPIClient()
    
    print("🧪 تست اتصال به ریپوی داده‌ها...")
    
    # تست لیست کوین‌ها
    coins = client.get_coins_list(limit=5, use_local=True)
    print(f"✅ لیست کوین‌ها: {len(coins.get('result', [])) if coins else 0} کوین")
    
    # تست داده‌های تحلیلی
    analytics = client.get_analytical_data(use_local=True)
    print(f"✅ داده‌های تحلیلی: {'✅' if analytics else '❌'}")
    
    # تست چارت کوین‌ها
    test_coins = ["bitcoin", "ethereum", "dogecoin"]
    for coin in test_coins:
        chart = client.get_coin_chart(coin, "1m", use_local=True)
        print(f"✅ چارت {coin}: {'✅' if chart else '❌'}")
    
    print("🎯 تست کامل شد!")

if __name__ == "__main__":
    test_remote_data()
