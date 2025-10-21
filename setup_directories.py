# setup_directories.py
import os
import json

def setup_project_structure():
    """ایجاد ساختار پوشه‌های پروژه"""
    
    # پوشه‌های اصلی
    directories = [
        'models/shared',
        'models/data/historical',
        'models/data/analysis',
        'models/data/models',
        'models/data/snapshots'
    ]
    
    # ایجاد پوشه‌ها
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ پوشه ایجاد شد: {directory}")
        except Exception as e:
            print(f"⚠️ خطا در ایجاد {directory}: {e}")
    
    # فایل‌های اولیه
    initial_files = {
        'models/shared/realtime_prices.json': {"timestamp": 0, "realtime_data": {}},
        'models/data/analysis/.gitkeep': "",
        'models/data/historical/.gitkeep': "", 
        'models/data/models/.gitkeep': "",
        'models/data/snapshots/.gitkeep': ""
    }
    
    for file_path, content in initial_files.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if isinstance(content, dict):
                    json.dump(content, f, indent=2, ensure_ascii=False)
                else:
                    f.write(content)
            print(f"✅ فایل ایجاد شد: {file_path}")
        except Exception as e:
            print(f"⚠️ خطا در ایجاد {file_path}: {e}")
    
    print("🎯 ساختار پروژه آماده است!")

if __name__ == "__main__":
    setup_project_structure()
