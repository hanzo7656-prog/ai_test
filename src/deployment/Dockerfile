# 📁 deployment/Dockerfile

FROM python:3.9-slim

# تنظیمات محیط
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME=/app

# ایجاد دایرکتوری برنامه
WORKDIR $APP_HOME

# نصب وابستگی‌های سیستم
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# کپی فایل‌های requirements
COPY requirements.txt .

# نصب پکیج‌های پایتون
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد برنامه
COPY . .

# ایجاد دایرکتوری‌های لازم
RUN mkdir -p logs models data

# پورت اکسپوز
EXPOSE 8000

# کاربر غیر root
RUN useradd --create-home --shell /bin/bash app
USER app

# کامند اجرا
CMD ["python", "src/api/main.py"]
