// تابع برای فرمت‌بندی هوشمند اعداد
function formatSmartDisplay(usedMB, percent, totalMB, type) {
    if (type === 'ram') {
        if (percent < 1) {
            return {
                displayPercent: percent * 10, // اسکیل برای نمایش بهتر
                displayText: `${percent}% (${usedMB} MB)`,
                status: "عالی ✅",
                color: "#10B981"
            };
        } else if (percent < 30) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (${usedMB} MB / ${Math.round(totalMB)} MB)`,
                status: "بهینه 👍", 
                color: "#8B5CF6"
            };
        } else {
            return {
                displayPercent: percent,
                displayText: `${percent}% (${usedMB} MB) ⚠️`,
                status: "نیاز توجه",
                color: "#F59E0B"
            };
        }
    } else if (type === 'cpu') {
        if (percent < 10) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (سبک)`,
                status: "عالی ✅",
                color: "#10B981"
            };
        } else if (percent < 50) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (متوسط)`,
                status: "نرمال 🔄",
                color: "#8B5CF6"
            };
        } else {
            return {
                displayPercent: percent,
                displayText: `${percent}% (سنگین) ⚠️`,
                status: "مشغول",
                color: "#EF4444"
            };
        }
    }
}

// آپدیت تابع اصلی
async function getHealthData() {
    try {
        addLog('🔄 در حال دریافت اطلاعات واقعی سیستم...');
        
        const response = await fetch('/health');
        const data = await response.json();
        
        // فرمت‌بندی RAM
        const ramDisplay = formatSmartDisplay(
            data.ram_used_mb, 
            data.ram_percent, 
            data.total_ram_mb, 
            'ram'
        );
        
        // فرمت‌بندی CPU  
        const cpuDisplay = formatSmartDisplay(
            data.cpu_percent,
            data.cpu_percent,
            null,
            'cpu'
        );
        
        // بروزرسانی RAM در UI
        document.getElementById('ram-percent').textContent = `${data.ram_percent}%`;
        document.getElementById('ram-value').textContent = ramDisplay.displayText;
        setProgress(document.querySelector('.ram-progress'), ramDisplay.displayPercent);
        document.querySelector('.ram-progress').style.stroke = ramDisplay.color;
        
        // بروزرسانی CPU در UI
        document.getElementById('cpu-percent').textContent = `${data.cpu_percent}%`;
        document.getElementById('cpu-value').textContent = cpuDisplay.displayText;
        setProgress(document.querySelector('.cpu-progress'), data.cpu_percent);
        document.querySelector('.cpu-progress').style.stroke = cpuDisplay.color;
        
        // بروزرسانی نورون‌ها
        document.getElementById('neuron-count').textContent = data.neurons;
        
        // بروزرسانی وضعیت کلی
        document.getElementById('system-status').textContent = data.status;
        document.getElementById('system-status').style.background = 
            data.ram_percent > 50 || data.cpu_percent > 80 ? '#EF4444' : '#10B981';
        
        // بروزرسانی زمان
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString('fa-IR');
        
        addLog(`✅ RAM: ${ramDisplay.status} | CPU: ${cpuDisplay.status}`);
        
    } catch (error) {
        addLog('❌ خطا در دریافت اطلاعات سیستم');
        console.error('Error:', error);
    }
}

// تست پیش‌بینی با لاگ بهتر
async function testPrediction() {
    try {
        addLog('🧪 در حال تست پیش‌بینی AI...');
        
        const startTime = Date.now();
        const response = await fetch('/predict');
        const data = await response.json();
        const endTime = Date.now();
        
        addLog(`✅ پیش‌بینی: ${data.prediction} | زمان: ${endTime - startTime}ms`);
        
    } catch (error) {
        addLog('❌ خطا در تست پیش‌بینی');
        console.error('Error:', error);
    }
}
