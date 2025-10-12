// تنظیم نوارهای دایره‌ای
function setProgress(circle, percent) {
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percent / 100) * circumference;
    
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = offset;
}


// فرمت‌بندی هوشمند - فقط CPU رو اصلاح کن
function formatSmartDisplay(usedMB, percent, totalMB, type) {
    if (type === 'ram') {
        // RAM همون قبلی که کار می‌کنه
        if (percent < 30) {
            return {
                displayPercent: percent,
                displayText: `${usedMB} MB از ${totalMB} MB`,
                status: "بهینه ✅",
                statusClass: "status-excellent",
                color: "#10B981"
            };
        } else if (percent < 70) {
            return {
                displayPercent: percent,
                displayText: `${usedMB} MB از ${totalMB} MB`,
                status: "نرمال 👍", 
                statusClass: "status-optimal",
                color: "#8B5CF6"
            };
        } else {
            return {
                displayPercent: percent,
                displayText: `${usedMB} MB از ${totalMB} MB ⚠️`,
                status: "نیاز توجه",
                statusClass: "status-warning",
                color: "#F59E0B"
            };
        }
    } else if (type === 'cpu') {
        // CPU: نمایش مقادیر منطقی
        if (percent < 2) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (بسیار سبک)`,
                status: "عالی ✅",
                statusClass: "status-excellent",
                color: "#10B981"
            };
        } else if (percent < 5) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (سبک)`,
                status: "بهینه 👍",
                statusClass: "status-optimal",
                color: "#8B5CF6"
            };
        } else if (percent < 15) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (متوسط)`,
                status: "نرمال 🔄",
                statusClass: "status-optimal", 
                color: "#8B5CF6"
            };
        } else {
            return {
                displayPercent: percent,
                displayText: `${percent}% (سنگین) ⚠️`,
                status: "مشغول",
                statusClass: "status-warning",
                color: "#EF4444"
            };
        }
    }
}

// بروزرسانی دشبورد
async function getHealthData() {
    try {
        addLog('🔄 بروزرسانی وضعیت...');
        
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
        
        // بروزرسانی RAM
        document.getElementById('ram-percent').textContent = `${data.ram_percent}%`;
        document.getElementById('ram-value').textContent = ramDisplay.displayText;
        setProgress(document.querySelector('.ram-progress'), data.ram_percent);
        document.querySelector('.ram-progress').style.stroke = ramDisplay.color;
        
        // وضعیت RAM
        updateStatusIndicator('ram', ramDisplay.status, ramDisplay.statusClass);
        
        // بروزرسانی CPU
        document.getElementById('cpu-percent').textContent = `${data.cpu_percent}%`;
        document.getElementById('cpu-value').textContent = cpuDisplay.displayText;
        setProgress(document.querySelector('.cpu-progress'), data.cpu_percent);
        document.querySelector('.cpu-progress').style.stroke = cpuDisplay.color;
        
        // وضعیت CPU
        updateStatusIndicator('cpu', cpuDisplay.status, cpuDisplay.statusClass);
        
        // نورون‌ها
        document.getElementById('neuron-count').textContent = data.neurons;
        
        // وضعیت کلی
        document.getElementById('system-status').textContent = data.status;
        document.getElementById('system-status').style.background = 
            data.ram_percent > 70 || data.cpu_percent > 80 ? '#EF4444' : '#10B981';
        
        // زمان
        document.getElementById('last-update').textContent = data.server_time || new Date().toLocaleTimeString('fa-IR');
        
        addLog(`✅ RAM: ${data.ram_percent}% | CPU: ${data.cpu_percent}%`);
        
    } catch (error) {
        addLog('❌ خطا در دریافت اطلاعات');
        console.error('Error:', error);
    }
}

// آپدیت وضعیت
function updateStatusIndicator(type, status, statusClass) {
    let statusElement = document.getElementById(`${type}-status`);
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = `${type}-status`;
        statusElement.className = 'status-indicator';
        document.querySelector(`.${type}-progress`).parentNode.appendChild(statusElement);
    }
    statusElement.textContent = status;
    statusElement.className = `status-indicator ${statusClass}`;
}

// تست‌های مختلف
async function testPrediction() {
    try {
        addLog('🧪 تست پیش‌بینی AI...');
        const response = await fetch('/predict');
        const data = await response.json();
        addLog(`✅ ${data.prediction} | زمان: ${data.processing_time_ms}ms`);
        setTimeout(getHealthData, 1000);
    } catch (error) {
        addLog('❌ خطا در تست پیش‌بینی');
    }
}

async function testHeavyCPU() {
    try {
        addLog('⚡ تست CPU سنگین...');
        const response = await fetch('/test-cpu');
        const data = await response.json();
        addLog(`✅ تست CPU | زمان: ${data.processing_time_ms}ms`);
        setTimeout(getHealthData, 2000);
    } catch (error) {
        addLog('❌ خطا در تست CPU');
    }
}

async function testLightCPU() {
    try {
        addLog('🔵 تست CPU سبک...');
        const response = await fetch('/light-cpu');
        const data = await response.json();
        addLog(`✅ تست سبک | زمان: ${data.processing_time_ms}ms`);
        setTimeout(getHealthData, 1000);
    } catch (error) {
        addLog('❌ خطا در تست سبک');
    }
}

// لاگ
function addLog(message) {
    const logContent = document.getElementById('live-log');
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = `[${new Date().toLocaleTimeString('fa-IR')}] ${message}`;
    logContent.appendChild(logEntry);
    logContent.scrollTop = logContent.scrollHeight;
}

// مدیریت رویدادها
function setupEventListeners() {
    // دکمه‌های موجود
    document.querySelector('.btn-primary')?.addEventListener('click', getHealthData);
    document.querySelector('.btn-secondary')?.addEventListener('click', testPrediction);
    
    // اضافه کردن دکمه‌های تست CPU
    const controlPanel = document.querySelector('.control-panel');
    if (controlPanel) {
        if (!document.querySelector('.btn-cpu-heavy')) {
            const heavyBtn = document.createElement('button');
            heavyBtn.className = 'btn btn-warning btn-cpu-heavy';
            heavyBtn.innerHTML = '⚡ تست سنگین';
            heavyBtn.addEventListener('click', testHeavyCPU);
            controlPanel.appendChild(heavyBtn);
        }
        
        if (!document.querySelector('.btn-cpu-light')) {
            const lightBtn = document.createElement('button');
            lightBtn.className = 'btn btn-info btn-cpu-light';
            lightBtn.innerHTML = '🔵 تست سبک';
            lightBtn.addEventListener('click', testLightCPU);
            controlPanel.appendChild(lightBtn);
        }
    }
}

// مقداردهی اولیه
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    getHealthData();
    setInterval(getHealthData, 8000); // هر 8 ثانیه
    addLog('🚀 دشبورد AI راه‌اندازی شد');
});
