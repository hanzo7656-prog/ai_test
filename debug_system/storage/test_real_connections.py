#!/usr/bin/env python3
"""
تست واقعی اتصال به دیتابیس‌های Redis
"""

import sys
import os
import time
import json
from datetime import datetime

# ========== تنظیم مسیرها ==========
# اضافه کردن پوشه والد به مسیر
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print("🔧 تنظیمات اولیه...")
print(f"مسیر فعلی: {current_dir}")
print(f"مسیر والد: {parent_dir}")
print("-" * 50)

# ========== ایمپورت ماژول‌ها ==========
print("\n📦 در حال ایمپورت ماژول‌ها...")
try:
    from debug_system.storage.redis_manager import redis_manager
    print("✅ redis_manager ایمپورت شد")
except ImportError as e:
    print(f"❌ خطا در ایمپورت redis_manager: {e}")
    print("\nراه‌حل: مطمئن شوید که:")
    print("1. فایل redis_manager.py در همین پوشه وجود دارد")
    print("2. متغیرهای محیطی (.env) تنظیم شده‌اند")
    print("3. پکیج redis نصب شده: pip install redis")
    sys.exit(1)

# ========== تابع کمکی برای نمایش ==========
def print_section(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def print_test_result(operation, success, time_taken=None, details=""):
    icon = "✅" if success else "❌"
    time_str = f" [{time_taken:.3f}s]" if time_taken is not None else ""
    print(f"  {icon} {operation}{time_str} {details}")

# ========== ۱. تست اولیه: لیست دیتابیس‌ها ==========
print_section("۱. بررسی دیتابیس‌های موجود")

print("📋 لیست دیتابیس‌ها:")
for i, (db_name, client) in enumerate(redis_manager.databases.items(), 1):
    status = "🟢 فعال" if client else "🔴 غیرفعال"
    print(f"  {i}. {db_name}: {status}")

# ========== ۲. تست PING ساده ==========
print_section("۲. تست PING (ساده)")

for db_name, client in redis_manager.databases.items():
    print(f"\n📡 {db_name.upper()}:")
    
    if client is None:
        print("  ❌ کلاینت None است - احتمالاً متغیر محیطی تنظیم نشده")
        continue
    
    try:
        start = time.time()
        result = client.ping()
        response_time = (time.time() - start) * 1000  # به میلی‌ثانیه
        
        if result:
            print(f"  ✅ PONG! - {response_time:.1f}ms")
        else:
            print(f"  ❌ No response")
            
    except Exception as e:
        print(f"  ❌ خطا: {type(e).__name__}: {str(e)[:80]}")

# ========== ۳. تست عملیات واقعی SET/GET ==========
print_section("۳. تست عملیات واقعی (SET/GET/DELETE)")

# داده تست
test_payload = {
    "test": "این یک تست اتصال است",
    "timestamp": datetime.now().isoformat(),
    "number": 42,
    "list": [1, 2, 3],
    "nested": {"key": "value"}
}

for db_name in ['uta', 'utb', 'utc', 'mother_a', 'mother_b']:
    print(f"\n🔬 {db_name.upper()}:")
    
    # کلید تست
    test_key = f"test:{db_name}:{int(time.time())}"
    
    # ۱. SET
    try:
        success, set_time = redis_manager.set(db_name, test_key, test_payload, expire=30)
        print_test_result("SET", success, set_time, 
                         f"key={test_key}" if success else "")
        
        if not success:
            continue
            
        # ۲. GET
        data, get_time = redis_manager.get(db_name, test_key)
        if data is not None:
            # بررسی یکسان بودن داده
            is_valid = (data["test"] == test_payload["test"] and 
                       data["number"] == test_payload["number"])
            print_test_result("GET", True, get_time, 
                            f"داده معتبر: {is_valid}")
        else:
            print_test_result("GET", False, get_time, "داده null برگردانده شد")
            continue
            
        # ۳. EXISTS
        exists, exists_time = redis_manager.exists(db_name, test_key)
        print_test_result("EXISTS", exists, exists_time)
        
        # ۴. DELETE
        deleted, delete_time = redis_manager.delete(db_name, test_key)
        print_test_result("DELETE", deleted, delete_time)
        
        # ۵. تأیید حذف
        still_exists, _ = redis_manager.exists(db_name, test_key)
        if not still_exists:
            print_test_result("VERIFY", True, None, "کلید با موفقیت حذف شد")
        else:
            print_test_result("VERIFY", False, None, "کلید هنوز وجود دارد!")
            
        print(f"    🎯 {db_name}: تمام تست‌ها PASS شد!")
        
    except Exception as e:
        print(f"  ❌ خطا در تست: {type(e).__name__}: {str(e)[:100]}")

# ========== ۴. بررسی Health Check واقعی ==========
print_section("۴. Health Check کامل")

try:
    health = redis_manager.health_check()
    print("📊 گزارش سلامت:")
    
    connected_count = 0
    for db_name, status in health.items():
        if isinstance(status, dict):
            if status.get('status') == 'connected':
                connected_count += 1
                color = "🟢"
            else:
                color = "🔴"
                
            print(f"\n{color} {db_name.upper()}:")
            print(f"   وضعیت: {status.get('status', 'unknown')}")
            print(f"   Ping: {status.get('ping_time_ms', 0)}ms")
            print(f"   حافظه: {status.get('used_memory_mb', 0)}/{status.get('max_memory_mb', 0)}MB")
            print(f"   کلیدها: {status.get('keys', 'N/A')}")
        else:
            print(f"\n🔴 {db_name.upper()}: {status}")
    
    print(f"\n📈 خلاصه: {connected_count}/5 دیتابیس متصل")
    
except Exception as e:
    print(f"❌ خطا در health check: {e}")

# ========== ۵. تست کلیدهای موجود ==========
print_section("۵. بررسی کلیدهای موجود")

for db_name in ['utb', 'utc', 'mother_b']:  # مهم‌ترین‌ها
    try:
        keys, scan_time = redis_manager.get_keys(db_name, "*")
        if keys:
            print(f"\n🗝️ {db_name.upper()}: {len(keys)} کلید ({scan_time:.2f}s)")
            
            # نمایش چند کلید نمونه
            sample = keys[:5]
            for i, key in enumerate(sample):
                # بررسی TTL
                try:
                    ttl = redis_manager.get_client(db_name).ttl(key)
                    ttl_str = f" (TTL: {ttl}s)" if ttl > 0 else " (بدون TTL)"
                except:
                    ttl_str = ""
                    
                print(f"   {i+1}. {key[:50]}{'...' if len(key) > 50 else ''}{ttl_str}")
        else:
            print(f"\n📭 {db_name.upper()}: هیچ کلیدی وجود ندارد")
            
    except Exception as e:
        print(f"\n❌ {db_name.upper()}: خطا در اسکن کلیدها - {str(e)[:50]}")

# ========== ۶. تست عملکرد ==========
print_section("۶. تست عملکرد")

def performance_test():
    """تست سرعت عملیات"""
    print("\n⚡ تست سرعت ۱۰ عملیان سریع:")
    
    db = 'mother_b'  # برای تست عملکرد
    times = []
    
    for i in range(10):
        key = f"perf_test_{i}_{int(time.time())}"
        value = {"i": i, "time": time.time()}
        
        try:
            # SET
            start = time.time()
            redis_manager.set(db, key, value, 10)
            set_time = time.time() - start
            
            # GET
            start = time.time()
            redis_manager.get(db, key)
            get_time = time.time() - start
            
            # DELETE
            start = time.time()
            redis_manager.delete(db, key)
            delete_time = time.time() - start
            
            total = set_time + get_time + delete_time
            times.append(total)
            
            if i < 3:  # فقط ۳ تا اول رو نمایش بده
                print(f"  عملیات {i+1}: SET={set_time*1000:.1f}ms, "
                      f"GET={get_time*1000:.1f}ms, "
                      f"DELETE={delete_time*1000:.1f}ms")
                      
        except Exception:
            continue
    
    if times:
        avg = sum(times) / len(times) * 1000  # به میلی‌ثانیه
        print(f"\n📊 میانگین زمان هر عملیات: {avg:.1f}ms")
        
        if avg < 10:
            print("  🚀 عملکرد: عالی")
        elif avg < 50:
            print("  ⚡ عملکرد: خوب")
        elif avg < 100:
            print("  ⚠️  عملکرد: متوسط")
        else:
            print("  🐌 عملکرد: کند")
    else:
        print("❌ تست عملکرد ناموفق بود")

performance_test()

# ========== ۷. تست اتصال به متغیرهای محیطی ==========
print_section("۷. بررسی متغیرهای محیطی")

env_vars = [
    "UTA_REDIS_AI",
    "UTB_REDIS_AI", 
    "UTC_REDIS_AI",
    "MOTHER_A_URL",
    "MOTHER_B_URL"
]

print("\n🔍 بررسی متغیرهای محیطی Redis:")
for var in env_vars:
    value = os.getenv(var)
    if value:
        # مخفی کردن پسورد برای امنیت
        if "@" in value:
            # نمایش فقط host و port
            parts = value.split("@")
            if len(parts) == 2:
                safe_value = f"redis://***@{parts[1][:30]}..."
            else:
                safe_value = "redis://***:****@..."
        else:
            safe_value = value[:30] + "..." if len(value) > 30 else value
            
        print(f"  ✅ {var}: {safe_value}")
    else:
        print(f"  ❌ {var}: تنظیم نشده!")

# ========== نتیجه نهایی ==========
print_section("نتیجه نهایی")

print("\n🎯 جمع‌بندی:")
print("1. تست PING: بررسی پاسخ سریع")
print("2. تست SET/GET: بررسی عملیات ذخیره/بازیابی")
print("3. Health Check: بررسی وضعیت کامل")
print("4. بررسی کلیدها: مشاهده داده‌های موجود")
print("5. تست عملکرد: اندازه‌گیری سرعت")
print("6. بررسی env vars: اطمینان از تنظیمات")

print("\n📋 برای اجرای مجدد:")
print("python debug_system/storage/test_real_connections.py")

print("\n" + "="*60)
print("✅ تست کامل شد! لطفا نتایج بالا را بررسی کنید.")
print("="*60)
