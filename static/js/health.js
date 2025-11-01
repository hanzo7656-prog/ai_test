// static/js/health.js
class HealthMonitor {
    constructor() {
        this.initializeCharts();
        this.loadServicesStatus();
        this.loadAlerts();
        this.loadSystemLogs();
        this.startRealTimeUpdates();
    }

    initializeCharts() {
        // شبیه‌سازی نمودارهای سلامت
        this.createSimpleChart('cpuChart', [30, 25, 40, 35, 45, 30, 25], '#13bcff');
        this.createSimpleChart('memoryChart', [65, 70, 68, 72, 75, 70, 67], '#8b5cf6');
    }

    createSimpleChart(containerId, data, color) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // ایجاد یک نمودار ساده با CSS
        container.innerHTML = '';
        const chart = document.createElement('div');
        chart.className = 'simple-chart';
        
        data.forEach((value, index) => {
            const bar = document.createElement('div');
            bar.className = 'chart-bar';
            bar.style.height = `${value}%`;
            bar.style.backgroundColor = color;
            bar.style.opacity = 0.7 + (index * 0.05);
            chart.appendChild(bar);
        });
        
        container.appendChild(chart);
    }

    async loadServicesStatus() {
        const services = [
            {
                name: 'API CoinStats',
                description: 'اتصال به داده‌های بازار',
                status: 'healthy',
                icon: '🌐',
                latency: '142ms'
            },
            {
                name: 'هوش مصنوعی',
                description: 'مدل تحلیل پیشرفته',
                status: 'healthy', 
                icon: '🤖',
                accuracy: '87%'
            },
            {
                name: 'WebSocket',
                description: 'داده‌های Real-time',
                status: 'warning',
                icon: '⚡',
                message: 'اتصال ناپایدار'
            },
            {
                name: 'پایگاه داده',
                description: 'ذخیره‌سازی داده‌ها',
                status: 'healthy',
                icon: '💾',
                size: '2.4GB'
            },
            {
                name: 'Cache System',
                description: 'سیستم کش‌ینگ',
                status: 'critical',
                icon: '🚨',
                message: 'مصرف حافظه بالا'
            }
        ];

        this.renderServices(services);
    }

    renderServices(services) {
        const container = document.getElementById('servicesList');
        if (!container) return;

        container.innerHTML = services.map(service => `
            <div class="service-item ${service.status}">
                <div class="service-info">
                    <div class="service-icon">${service.icon}</div>
                    <div class="service-details">
                        <h4>${service.name}</h4>
                        <div class="service-desc">${service.description}</div>
                    </div>
                </div>
                <div class="service-status">
                    <span class="status-badge ${service.status}">
                        ${this.getStatusText(service.status)}
                    </span>
                    ${service.latency ? `<span class="latency">${service.latency}</span>` : ''}
                </div>
            </div>
        `).join('');
    }

    getStatusText(status) {
        const statusMap = {
            healthy: 'سالم',
            warning: 'هشدار', 
            critical: 'بحرانی'
        };
        return statusMap[status] || status;
    }

    async loadAlerts() {
        const alerts = [
            {
                type: 'critical',
                icon: '🚨',
                title: 'مصرف حافظه بحرانی',
                description: 'مصرف حافظه به ۸۵٪ رسیده است',
                time: '۲ دقیقه پیش',
                actions: true
            },
            {
                type: 'warning', 
                icon: '⚠️',
                title: 'اتصال WebSocket ناپایدار',
                description: 'اتصال با وقفه مواجه شده است',
                time: '۵ دقیقه پیش',
                actions: true
            },
            {
                type: 'info',
                icon: 'ℹ️',
                title: 'بروزرسانی مدل AI',
                description: 'مدل در حال آموزش است',
                time: '۱۰ دقیقه پیش', 
                actions: false
            }
        ];

        this.renderAlerts(alerts);
    }

    renderAlerts(alerts) {
        const container = document.getElementById('healthAlertsList');
        const countElement = document.getElementById('alertsCount');
        
        if (!container) return;

        const criticalCount = alerts.filter(alert => alert.type === 'critical').length;
        if (countElement) {
            countElement.textContent = criticalCount;
        }

        container.innerHTML = alerts.map(alert => `
            <div class="alert-item ${alert.type}">
                <div class="alert-icon">${alert.icon}</div>
                <div class="alert-content">
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-desc">${alert.description}</div>
                    <div class="alert-time">${alert.time}</div>
                </div>
                ${alert.actions ? `
                    <div class="alert-actions">
                        <button class="btn btn-secondary" onclick="this.resolveAlert('${alert.title}')">
                            رفع
                        </button>
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    async loadSystemLogs() {
        const logs = [
            {
                level: 'info',
                time: '۱۴:۳۰:۲۵',
                message: 'سیستم با موفقیت راه‌اندازی شد'
            },
            {
                level: 'info', 
                time: '۱۴:۳۱:۱۰',
                message: 'اتصال به API CoinStats برقرار شد'
            },
            {
                level: 'warning',
                time: '۱۴:۳۲:۴۵', 
                message: 'تأخیر در پاسخ API - ۳۴۲ms'
            },
            {
                level: 'error',
                time: '۱۴:۳۳:۲۰',
                message: 'خطا در دریافت داده‌های ETH/USDT'
            },
            {
                level: 'info',
                time: '۱۴:۳۴:۱۵',
                message: 'بروزرسانی کش انجام شد'
            }
        ];

        this.renderLogs(logs);
    }

    renderLogs(logs) {
        const container = document.getElementById('systemLogs');
        if (!container) return;

        container.innerHTML = logs.map(log => `
            <div class="log-entry">
                <span class="log-level ${log.level}">${log.level.toUpperCase()}</span>
                <span class="log-time">${log.time}</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');
    }

    startRealTimeUpdates() {
        // بروزرسانی Real-time متریک‌ها
        setInterval(() => {
            this.updateMetrics();
        }, 5000);

        // بروزرسانی نمودارها
        setInterval(() => {
            this.updateCharts();
        }, 10000);
    }

    updateMetrics() {
        // شبیه‌سازی بروزرسانی متریک‌ها
        const cpu = 20 + Math.random() * 30;
        const memory = 60 + Math.random() * 20;
        const latency = 100 + Math.random() * 100;
        const accuracy = 85 + Math.random() * 10;

        document.getElementById('cpuUsage').textContent = `${Math.round(cpu)}٪`;
        document.getElementById('memoryUsage').textContent = `${Math.round(memory)}٪`;
        document.getElementById('apiLatency').textContent = `${Math.round(latency)}ms`;
        document.getElementById('aiAccuracy').textContent = `${Math.round(accuracy)}٪`;

        // آپدیت progress bar
        this.updateProgressBar('cpuUsage', cpu);
        this.updateProgressBar('memoryUsage', memory);
        this.updateProgressBar('apiLatency', latency / 5); // نرمال‌سازی
        this.updateProgressBar('aiAccuracy', accuracy);
    }

    updateProgressBar(metricId, value) {
        const card = document.getElementById(metricId).closest('.metric-card');
        const progressFill = card.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.width = `${Math.min(value, 100)}%`;
        }
    }

    updateCharts() {
        // بروزرسانی داده‌های نمودار
        const newCpuData = Array.from({length: 7}, () => 20 + Math.random() * 40);
        const newMemoryData = Array.from({length: 7}, () => 60 + Math.random() * 25);

        this.createSimpleChart('cpuChart', newCpuData, '#13bcff');
        this.createSimpleChart('memoryChart', newMemoryData, '#8b5cf6');
    }

    resolveAlert(alertTitle) {
        // شبیه‌سازی رفع هشدار
        console.log(`رفع هشدار: ${alertTitle}`);
        alert(`هشدار "${alertTitle}" رفع شد`);
    }
}

// راه‌اندازی زمانی که DOM لود شد
document.addEventListener('DOMContentLoaded', () => {
    new HealthMonitor();

    // مدیریت فیلترها
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            const filter = this.dataset.filter;
            // اینجا منطق فیلتر کردن پیاده‌سازی میشه
        });
    });

    // مدیریت کنترل‌های لاگ
    document.getElementById('refreshLogs')?.addEventListener('click', () => {
        new HealthMonitor().loadSystemLogs();
    });

    document.getElementById('clearLogs')?.addEventListener('click', () => {
        document.getElementById('systemLogs').innerHTML = '<div class="log-entry">لاگ‌ها پاک شدند</div>';
    });
});
