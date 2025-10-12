// تنظیم نوارهای دایره‌ای
function setProgress(circle, percent) {
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percent / 100) * circumference;
    
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = offset;
}

// بروزرسانی دشبورد
async function getHealthData() {
    try {
        addLog('🔄 در حال دریافت اطلاعات سیستم...');
        
        const response = await fetch('/health');
        const data = await response.json();
        
        // بروزرسانی RAM
        document.getElementById('ram-percent').textContent = `${data.ram_percent}%`;
        document.getElementById('ram-value').textContent = `${data.ram_used_mb} MB`;
        setProgress(document.querySelector('.ram-progress'), data.ram_percent);
        
        // بروزرسانی CPU
        document.getElementById('cpu-percent').textContent = `${data.cpu_percent}%`;
        document.getElementById('cpu-value').textContent = `${data.cpu_percent}%`;
        setProgress(document.querySelector('.cpu-progress'), data.cpu_percent);
        
        // بروزرسانی نورون‌ها
        document.getElementById('neuron-count').textContent = data.neurons;
        
        // بروزرسانی وضعیت سیستم
        document.getElementById('system-status').textContent = data.status;
        document.getElementById('system-status').className = 'value status-badge';
        document.getElementById('system-status').style.background = 
            data.ram_percent > 80 ? '#EF4444' : data.ram_percent > 60 ? '#F59E0B' : '#10B981';
        
        // بروزرسانی زمان
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString('fa-IR');
        
        addLog('✅ اطلاعات سیستم با موفقیت بروزرسانی شد');
        
    } catch (error) {
        addLog('❌ خطا در دریافت اطلاعات سیستم');
        console.error('Error:', error);
    }
}

// تست پیش‌بینی
async function testPrediction() {
    try {
        addLog('🧪 در حال تست پیش‌بینی AI...');
        
        const response = await fetch('/predict');
        const data = await response.json();
        
        addLog(`✅ پیش‌بینی: ${data.prediction} | زمان: ${data.processing_time_ms}ms`);
        
    } catch (error) {
        addLog('❌ خطا در تست پیش‌بینی');
        console.error('Error:', error);
    }
}

// اضافه کردن لاگ
function addLog(message) {
    const logContent = document.getElementById('live-log');
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = `[${new Date().toLocaleTimeString('fa-IR')}] ${message}`;
    
    logContent.appendChild(logEntry);
    logContent.scrollTop = logContent.scrollHeight;
}

// بروزرسانی خودکار هر 10 ثانیه
setInterval(getHealthData, 10000);

// مقداردهی اولیه
document.addEventListener('DOMContentLoaded', function() {
    getHealthData();
    addLog('🚀 دشبورد AI راه‌اندازی شد');
});
