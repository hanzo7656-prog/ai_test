// توابع اصلی سیستم
function updateSystemInfo() {
    fetch('/health')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('📊 داده‌های دریافتی:', data);
            
            // آپدیت RAM
            const ramPercent = Math.min(data.ram_percent || 0, 100);
            const ramValue = data.ram_used_mb || 0;
            
            document.getElementById('ram-percent').textContent = ramPercent + '%';
            document.getElementById('ram-value').textContent = ramValue.toFixed(1) + ' MB';
            updateProgressCircle('.ram-progress', ramPercent);
            
            // آپدیت CPU
            const cpuPercent = Math.min(data.cpu_percent || 0, 100);
            
            document.getElementById('cpu-percent').textContent = cpuPercent + '%';
            document.getElementById('cpu-value').textContent = cpuPercent + '%';
            updateProgressCircle('.cpu-progress', cpuPercent);
            
            // آپدیت نورون‌ها و وضعیت
            document.getElementById('neuron-count').textContent = data.neurons || 100;
            document.getElementById('system-status').textContent = data.status || 'فعال';
            
            // آپدیت زمان
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString('fa-IR');
            
            addLog('✅ وضعیت سیستم بروزرسانی شد');
            
        })
        .catch(error => {
            console.error('❌ خطا در دریافت داده:', error);
            document.getElementById('system-status').textContent = 'اتصال قطع';
            document.getElementById('last-update').textContent = 'خطا';
            addLog('❌ خطا در بروزرسانی وضعیت');
        });
}

function updateProgressCircle(selector, percent) {
    const circle = document.querySelector(selector);
    if (circle) {
        const circumference = 2 * Math.PI * 54;
        const offset = circumference - (percent / 100) * circumference;
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = offset;
    }
}

// توابع هوش مصنوعی پیشرفته
async function testAIConnection() {
    document.getElementById('ai-output').textContent = '⏳ در حال بررسی سلامت هوش مصنوعی...';
    try {
        const response = await fetch('/health');
        const data = await response.json();
        document.getElementById('ai-output').textContent = JSON.stringify(data, null, 2);
        addLog('✅ تست سلامت هوش مصنوعی موفق');
    } catch (error) {
        document.getElementById('ai-output').textContent = '❌ خطا در تست سلامت: ' + error.message;
        addLog('❌ خطا در تست سلامت هوش مصنوعی');
    }
}

async function predictMarket() {
    document.getElementById('ai-output').textContent = '⏳ در حال پیش‌بینی بازار...';
    try {
        const response = await fetch('/predict/market');
        const data = await response.json();
        document.getElementById('ai-output').textContent = JSON.stringify(data, null, 2);
        addLog('📈 پیش‌بینی بازار انجام شد');
    } catch (error) {
        document.getElementById('ai-output').textContent = '❌ خطا در پیش‌بینی بازار: ' + error.message;
        addLog('❌ خطا در پیش‌بینی بازار');
    }
}

async function analyzeBTC() {
    document.getElementById('ai-output').textContent = '⏳ در حال تحلیل بیت‌کوین...';
    try {
        const response = await fetch('/analyze/coin/BTC');
        const data = await response.json();
        document.getElementById('ai-output').textContent = JSON.stringify(data, null, 2);
        addLog('🔍 تحلیل بیت‌کوین انجام شد');
    } catch (error) {
        document.getElementById('ai-output').textContent = '❌ خطا در تحلیل بیت‌کوین: ' + error.message;
        addLog('❌ خطا در تحلیل بیت‌کوین');
    }
}

async function systemForecast() {
    document.getElementById('ai-output').textContent = '⏳ در حال پیش‌بینی منابع سیستم...';
    try {
        const response = await fetch('/system/forecast');
        const data = await response.json();
        document.getElementById('ai-output').textContent = JSON.stringify(data, null, 2);
        addLog('🔮 پیش‌بینی منابع سیستم انجام شد');
    } catch (error) {
        document.getElementById('ai-output').textContent = '❌ خطا در پیش‌بینی منابع: ' + error.message;
        addLog('❌ خطا در پیش‌بینی منابع سیستم');
    }
}

async function testMiddleware() {
    document.getElementById('ai-output').textContent = '⏳ در حال تست اتصال به سرور میانی...';
    try {
        const response = await fetch('/test/middleware-connection');
        const data = await response.json();
        document.getElementById('ai-output').textContent = JSON.stringify(data, null, 2);
        addLog('🌐 تست اتصال سرور میانی انجام شد');
    } catch (error) {
        document.getElementById('ai-output').textContent = '❌ خطا در تست اتصال: ' + error.message;
        addLog('❌ خطا در تست اتصال سرور میانی');
    }
}

// توابع کمکی
function addLog(message) {
    const logElement = document.getElementById('live-log');
    const timestamp = new Date().toLocaleTimeString('fa-IR');
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = `[${timestamp}] ${message}`;
    logElement.appendChild(logEntry);
    logElement.scrollTop = logElement.scrollHeight;
}

function testPrediction() {
    fetch('/predict')
        .then(response => response.json())
        .then(data => {
            addLog('🧠 تست پیش‌بینی: ' + (data.prediction || 'انجام شد'));
        })
        .catch(error => {
            addLog('❌ خطا در تست پیش‌بینی');
        });
}

function getHealthData() {
    updateSystemInfo();
}

// اجرای خودکار هنگام بارگذاری صفحه
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 صفحه بارگذاری شد - شروع به روزرسانی...');
    updateSystemInfo();
    
    // بروزرسانی خودکار هر 10 ثانیه
    setInterval(updateSystemInfo, 10000);
    
    // انیمیشن نورون‌ها
    setInterval(() => {
        const dots = document.querySelectorAll('.neuron-dot');
        dots.forEach(dot => {
            if (Math.random() > 0.7) {
                dot.style.opacity = Math.random() > 0.5 ? '1' : '0.3';
            }
        });
    }, 1000);
});
