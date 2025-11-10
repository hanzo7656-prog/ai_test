import os
import re

def add_cache_to_routes():
    """اضافه کردن خودکار دکوراتور کش به routes"""
    
    route_files = [
        "routes/coins.py",
        "routes/news.py", 
        "routes/insights.py",
        "routes/exchanges.py",
        "routes/raw_coins.py",
        "routes/raw_news.py",
        "routes/raw_insights.py",
        "routes/raw_exchanges.py"
    ]
    
    cache_imports = {
        "coins": "from debug_system.storage import cache_coins",
        "news": "from debug_system.storage import cache_news", 
        "insights": "from debug_system.storage import cache_insights",
        "exchanges": "from debug_system.storage import cache_exchanges"
    }
    
    cache_decorators = {
        "coins": "@cache_coins(expire=300)",
        "news": "@cache_news(expire=600)",
        "insights": "@cache_insights(expire=1800)", 
        "exchanges": "@cache_exchanges(expire=600)"
    }
    
    for file_path in route_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # تشخیص نوع route از نام فایل
        route_type = None
        for rt in cache_imports.keys():
            if rt in file_path:
                route_type = rt
                break
        
        if not route_type:
            continue
        
        # اضافه کردن import اگر وجود ندارد
        import_line = cache_imports[route_type]
        if import_line not in content:
            # پیدا کردن آخرین import و اضافه کردن بعد از آن
            import_pattern = r'(^import .*$|^from .* import .*$)'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            if imports:
                last_import = imports[-1]
                content = content.replace(last_import, f"{last_import}\n{import_line}")
        
        # اضافه کردن دکوراتور به توابع اصلی
        decorator = cache_decorators[route_type]
        
        # الگو برای پیدا کردن توابع اصلی
        function_pattern = r'(@router\.[a-z]+\(["\'].*["\']\)\s*\n)(async def [a-z_]+\(.*\):)'
        
        def add_decorator(match):
            return f"{match.group(1)}{decorator}\n{match.group(2)}"
        
        new_content = re.sub(function_pattern, add_decorator, content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Updated: {file_path}")
        else:
            print(f"⚠️ No changes: {file_path}")

if __name__ == "__main__":
    add_cache_to_routes()
    print("🎉 All routes updated with cache decorators!")
