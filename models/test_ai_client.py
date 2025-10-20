# test_ai_client.py
from AIClient import VortexAIClient
import asyncio

async def run_ai_client_tests():
    print("🚀 Starting Vortex AI Client Tests...\n")
    
    ai_client = VortexAIClient()
    
    try:
        # تست ۱: سلامت سرور
        print("1. Testing Server Health...")
        health = ai_client.test_server_health()
        print(f"✅ Server Health: {'HEALTHY' if health['success'] else 'UNHEALTHY'}")
        
        # تست ۲: تست کامل اتصال
        print("\n2. Running Comprehensive Connection Tests...")
        connection_test = ai_client.test_all_endpoints_comprehensive()
        summary = connection_test['summary']
        print(f"📊 Connection Test Summary: {summary['passed_tests']}/{summary['total_tests']} passed")
        
        # تست ۳: تست یکپارچگی داده‌های خام
        print("\n3. Testing Raw Data Integrity...")
        integrity_test = ai_client.test_raw_data_integrity()
        print(f"🔍 Data Integrity: {'PASSED' if integrity_test['success'] else 'FAILED'}")
        for result in integrity_test['results']:
            print(f"   {result['test']}: {'VALID' if result['valid'] else 'INVALID'}")
        
        # تست ۴: دریافت نمونه داده‌های تاریخی
        print("\n4. Testing Historical Data Endpoints...")
        historical_tests = [
            ('1H', ai_client.get_raw_historical_1h),
            ('24H', ai_client.get_raw_historical_24h), 
            ('7D', ai_client.get_raw_historical_7d)
        ]
        
        for period_name, method in historical_tests:
            result = method()
            print(f"   Historical {period_name}: {'SUCCESS' if result['success'] else 'FAILED'}")
            if result['success']:
                data = result['data']['raw_data'] if 'raw_data' in result['data'] else result['data']
                if hasattr(data, '__len__'):
                    print(f"     Data Points: {len(data)}")
                else:
                    print(f"     Data Type: {type(data).__name__}")
        
        # تست ۵: وضعیت نهایی
        print("\n5. Final Health Status...")
        final_health = ai_client.get_health_status()
        status = "ALL SYSTEMS GO" if final_health['server']['is_healthy'] else "SOME ISSUES"
        print(f"🏥 Final Status: {status}")
        print(f"📈 Endpoints Checked: {len(final_health['server']['endpoints'])}")
        
        print("\n🎉 AI Client Tests Completed!")
        
    except Exception as error:
        print(f"💥 Test Suite Failed: {error}")

if __name__ == "__main__":
    asyncio.run(run_ai_client_tests())
