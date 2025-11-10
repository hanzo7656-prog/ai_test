import os
import re

def add_cache_to_all_routes():
    """اضافه کردن خودکار دکوراتور کش به همه ۸ فایل route"""
    
    # 🔽 لیست کامل ۸ فایل route
    route_files = [
        # پردازش شده (۴ فایل)
        "routes/coins.py",
        "routes/news.py", 
        "routes/insights.py",
        "routes/exchanges.py",
        # خام (۴ فایل)
        "routes/raw_coins.py",
        "routes/raw_news.py",
        "routes/raw_insights.py",
        "routes/raw_exchanges.py"
    ]
    
    # 🔽 importهای مخصوص هر فایل
    cache_imports = {
        "coins.py": "from debug_system.storage import cache_coins",
        "news.py": "from debug_system.storage import cache_news", 
        "insights.py": "from debug_system.storage import cache_insights",
        "exchanges.py": "from debug_system.storage import cache_exchanges",
        "raw_coins.py": "from debug_system.storage import cache_raw_coins",
        "raw_news.py": "from debug_system.storage import cache_raw_news",
        "raw_insights.py": "from debug_system.storage import cache_raw_insights", 
        "raw_exchanges.py": "from debug_system.storage import cache_raw_exchanges"
    }
    
    # 🔽 دکوراتورهای مخصوص هر فایل با TTLهای مختلف
    cache_decorators = {
        # پردازش شده - TTL بیشتر
        "coins.py": "@cache_coins(expire=300)",           # ۵ دقیقه
        "news.py": "@cache_news(expire=600)",             # ۱۰ دقیقه  
        "insights.py": "@cache_insights(expire=1800)",    # ۳۰ دقیقه
        "exchanges.py": "@cache_exchanges(expire=600)",   # ۱۰ دقیقه
        
        # خام - TTL کمتر
        "raw_coins.py": "@cache_raw_coins(expire=180)",      # ۳ دقیقه
        "raw_news.py": "@cache_raw_news(expire=300)",        # ۵ دقیقه
        "raw_insights.py": "@cache_raw_insights(expire=900)", # ۱۵ دقیقه
        "raw_exchanges.py": "@cache_raw_exchanges(expire=300)" # ۵ دقیقه
    }
    
    for file_path in route_files:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        file_name = os.path.basename(file_path)
        route_type = cache_imports.get(file_name)
        decorator = cache_decorators.get(file_name)
        
        if not route_type or not decorator:
            print(f"❌ No config for: {file_name}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # اضافه کردن import اگر وجود ندارد
        if route_type not in content:
            # پیدا کردن آخرین import و اضافه کردن بعد از آن
            import_pattern = r'(^import .*$|^from .* import .*$)'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            if imports:
                last_import = imports[-1]
                content = content.replace(last_import, f"{last_import}\n{route_type}")
                print(f"✅ Added import to: {file_name}")
        
        # اضافه کردن دکوراتور به توابع اصلی
        # الگو برای پیدا کردن توابع اصلی (GET, POST, etc.)
        function_pattern = r'(@router\.[a-z]+\(["\'].*["\']\)\s*\n)(async def [a-z_]+\(.*\):)'
        
        def add_decorator(match):
            return f"{match.group(1)}{decorator}\n{match.group(2)}"
        
        new_content = re.sub(function_pattern, add_decorator, content)
        
        # شمارش تعداد توابع آپدیت شده
        original_functions = len(re.findall(function_pattern, content))
        updated_functions = len(re.findall(function_pattern, new_content))
        changes = original_functions - updated_functions == 0  # اگر تعداد توابع تغییر نکرده یعنی آپدیت شده
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # پیدا کردن تعداد دکوراتورهای اضافه شده
            decorator_count = new_content.count(decorator)
            print(f"✅ Updated: {file_name} ({decorator_count} endpoints cached)")
        else:
            # چک کن اگر از قبل دکوراتور وجود دارد
            existing_decorators = content.count(decorator.split('(')[0])
            if existing_decorators > 0:
                print(f"⚠️ Already cached: {file_name} ({existing_decorators} endpoints)")
            else:
                print(f"❌ No endpoints found: {file_name}")

if __name__ == "__main__":
    print("🔄 Adding cache decorators to all 8 route files...")
    add_cache_to_all_routes()
    print("🎉 All 8 route files updated with cache!")
    print("\n📊 Summary:")
    print("   • 4 processed routes: coins, news, insights, exchanges")
    print("   • 4 raw data routes: raw_coins, raw_news, raw_insights, raw_exchanges")
    print("   • Different TTLs for processed vs raw data")
