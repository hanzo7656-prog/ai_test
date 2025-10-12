// تنظیم نوارهای دایره‌ای
function setProgress(circle, percent) {
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percent / 100) * circumference;
    
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = offset;
}

// فرمت‌بندی هوشمند اعداد
function formatSmartDisplay(usedMB, percent, totalMB, type) {
    if (type === 'ram') {
        if (percent < 30) {
            return {
                displayPercent: percent,
                displayText: `${usedMB} MB (${percent}%)`,
                status: "بهینه",
                statusClass: "status-excellent"
            };
        } else if (percent < 70) {
            return {
                displayPercent: percent,
                displayText: `${usedMB} MB (${percent}%)`,
                status: "نرمال",
                statusClass: "status-optimal"
            };
        } else {
            return {
                displayPercent: percent,
                displayText: `${usedMB} MB (${percent}%) ⚠️`,
                status: "نیاز توجه",
                statusClass: "status-warning"
            };
        }
    } else if (type === 'cpu') {
        if (percent < 10) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (سبک)`,
                status: "عالی",
                statusClass: "status-excellent"
            };
        } else if (percent < 50) {
            return {
                displayPercent: percent,
                displayText: `${percent}% (متوسط)`,
                status: "نرمال",
                statusClass: "status-optimal"
            };
        } else {
            return {
                displayPercent: percent,
                displayText: `${percent}% (سنگین) ⚠️`,
                status: "مشغول",
                statusClass: "status-warning"
            };
        }
    }
}

// بروزرسانی دشبورد
async function getHealthData() {
    try {
        addLog('🔄 در حال دریافت اطلاعات سیستم...');
        
        const response = await fetch('/health');
        const data = await response.json();
        
        // فرمت‌بندی RAM
        const ramDisplay = formatSmartDisplay(
            data.ram_used_mb, 
            data.ram_percent, 
            data.total_ram_mb, 
            'ram'
        );
        
        // فرمت‌بندی CPU - اگر 0 بود مقدار پیش‌فرض بذار
        const cpuPercent = data.cpu_percent === 0 ? 0.5 : data.cpu_percent;
        const cpuDisplay = formatSmartDisplay(
            cpuPercent,
            cpuPercent,
            null,
            'cpu'
        );
        
        // بروزرسانی RAM در UI
        document.getElementById('ram-percent').textContent = `${data.ram_percent}%`;
        document.getElementById('ram-value').textContent = ramDisplay.displayText;
        setProgress(document.querySelector('.ram-progress'), data.ram_percent);
        
        // اضافه کردن وضعیت RAM
        let ramStatusElement = document.getElementById('ram-status');
        if (!ramStatusElement) {
            ramStatusElement = document.createElement('div');
            ramStatusElement.id = 'ram-status';
            ramStatusElement.className = 'status-indicator';
            document.querySelector('.ram-progress').parentNode.appendChild(ramStatusElement);
        }
        ramStatusElement.textContent = ramDisplay.status;
        ramStatusElement.className = `status-indicator ${ramDisplay.statusClass}`;
        
        // بروزرسانی CPU در UI
        document.getElementById('cpu-percent').textContent = `${cpuPercent}%`;
        document.getElementById('cpu-value').textContent = cpuDisplay.displayText;
        setProgress(document.querySelector('.cpu-progress'), cpuPercent);
        
        // اضافه کردن وضعیت CPU
        let cpuStatusElement = document.getElementById('cpu-status');
        if (!cpuStatusElement) {
            cpuStatusElement = document.createElement('div');
            cpuStatusElement.id = 'cpu-status';
            cpuStatusElement.className = 'status-indicator';
            document.querySelector('.cpu-progress').parentNode.appendChild(cpuStatusElement);
        }
        cpuStatusElement.textContent = cpuDisplay.status;
        cpuStatusElement.className = `status-indicator ${cpuDisplay.statusClass}`;
        
        // بروزرسانی نورون‌ها
        document.getElementById('neuron-count').textContent = data.neurons;
        
        // بروزرسانی وضعیت کلی
        document.getElementById('system-status').textContent = data.status;
        document.getElementById('system-status').style.background = 
            data.ram_percent > 70 || cpuPercent > 80 ? '#EF4444' : '#10B981';
        
        // بروزرسانی زمان
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString('fa-IR');
        
        addLog(`✅ RAM: ${data.ram_used_mb}MB (${data.ram_percent}%) | CPU: ${cpuPercent}%`);
        
    } catch (error) {
        addLog('❌ خطا در دریافت اطلاعات سیستم');
        console.error('Error:', error);
    }
}

// تست پیش‌بینی
async function testPrediction() {
    try {
        addLog('🧪 در حال تست پیش‌بینی AI...');
        
        const startTime = Date.now();
        const response = await fetch('/predict');
        const data = await response.json();
        const endTime = Date.now();
        
        addLog(`✅ پیش‌بینی: ${data.prediction} | زمان: ${endTime - startTime}ms`);
        
        // بعد از تست، وضعیت رو بروزرسانی کن
        setTimeout(getHealthData, 1000);
        
    } catch (error) {
        addLog('❌ خطا در تست پیش‌بینی');
        console.error('Error:', error);
    }
}

// تست CPU
async function testCPU() {
    try {
        addLog('⚡ در حال تست CPU...');
        
        const startTime = Date.now();
        const response = await fetch('/test-cpu');
        const data = await response.json();
        const endTime = Date.now();
        
        addLog(`✅ تست CPU انجام شد | زمان: ${data.processing_time_ms}ms`);
        
        // بعد از تست، وضعیت رو بروزرسانی کن
        setTimeout(getHealthData, 1500);
        
    } catch (error) {
        addLog('❌ خطا در تست CPU');
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

// مقداردهی اولیه نمودارها
function initializeCharts() {
    const ramCircle = document.querySelector('.ram-progress');
    const cpuCircle = document.querySelector('.cpu-progress');
    
    if (ramCircle) setProgress(ramCircle, 0);
    if (cpuCircle) setProgress(cpuCircle, 0);
}

// بروزرسانی خودکار هر 10 ثانیه
let autoRefreshInterval;

function startAutoRefresh() {
    autoRefreshInterval = setInterval(getHealthData, 10000);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
}

// مدیریت رویدادها
function setupEventListeners() {
    // دکمه بروزرسانی وضعیت
    const refreshBtn = document.querySelector('.btn-primary');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', getHealthData);
    }
    
    // دکمه تست پیش‌بینی
    const testBtn = document.querySelector('.btn-secondary');
    if (testBtn) {
        testBtn.addEventListener('click', testPrediction);
    }
    
    // اضافه کردن دکمه تست CPU
    const controlPanel = document.querySelector('.control-panel');
    if (controlPanel && !document.querySelector('.btn-cpu-test')) {
        const cpuTestBtn = document.createElement('button');
        cpuTestBtn.className = 'btn btn-secondary btn-cpu-test';
        cpuTestBtn.innerHTML = '⚡ تست CPU';
        cpuTestBtn.addEventListener('click', testCPU);
        controlPanel.appendChild(cpuTestBtn);
    }
}

// مقداردهی اولیه
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    getHealthData();
    startAutoRefresh();
    addLog('🚀 دشبورد AI راه‌اندازی شد');
});

// مدیریت when page becomes visible
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopAutoRefresh();
    } else {
        startAutoRefresh();
        getHealthData(); // بروزرسانی فوری وقتی صفحه visible میشه
    }
});
